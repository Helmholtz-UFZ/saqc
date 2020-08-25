#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODOS:
  - integrate plotting into the api
  - `data` and `flagger` as arguments to `getResult`
"""

import logging
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Callable

import pandas as pd
import dios
import numpy as np
import timeit

from saqc.lib.plotting import plotHook, plotAllHook
from saqc.lib.tools import isQuoted
from saqc.core.reader import readConfig
from saqc.flagger import BaseFlagger, CategoricalFlagger, SimpleFlagger, DmpFlagger
from saqc.core.register import FUNC_MAP


logger = logging.getLogger("SaQC")


def _handleErrors(exc, func_dump, policy):
    func_name = func_dump['func_name']
    func_kws = func_dump['func_kws']
    field = func_dump['field']
    ctrl_kws = func_dump['ctrl_kws']
    lineno = ctrl_kws['lineno']
    func_expr = ctrl_kws['lineno']
    msg = f"Execution failed. Variable: '{field}', "
    if lineno is not None and func_expr is not None:
        msg += f"Config line {lineno}: '{func_expr}', "
    else:
        msg += f"Function: {func_name}(), parameters: '{func_kws}', "
    msg += f"Exception:\n{type(exc).__name__}: {exc}"

    if policy == "ignore":
        logger.debug(msg)
    elif policy == "warn":
        logger.warning(msg)
    else:
        logger.error(msg)
        raise


def _prepInput(flagger, data, flags):
    dios_like = (dios.DictOfSeries, pd.DataFrame)

    if not isinstance(data, dios_like):
        raise TypeError("data must be of type dios.DictOfSeries or pd.DataFrame")

    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
            raise TypeError("data should not use MultiIndex")
        data = dios.to_dios(data)

    if not isinstance(flagger, BaseFlagger):
        # NOTE: we should generate that list automatically,
        #       it won't ever be complete otherwise
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f"flagger must be of type {flaggerlist} or a subclass of {BaseFlagger}")

    if flags is not None:
        if not isinstance(flags, dios_like):
            raise TypeError("flags must be of type dios.DictOfSeries or pd.DataFrame")

        if isinstance(flags, pd.DataFrame):
            if isinstance(flags.index, pd.MultiIndex) or isinstance(flags.columns, pd.MultiIndex):
                raise TypeError("flags' should not use MultiIndex")
            flags = dios.to_dios(flags)

        # NOTE: do not test all columns as they not necessarily need to be the same
        cols = flags.columns & data.columns
        if not (flags[cols].lengths == data[cols].lengths).all():
            raise ValueError("the length of flags and data need to be equal")

    if flagger.initialized:
        if not data.columns.difference(flagger.getFlags().columns).empty:
            raise ValueError("Given flagger does not contain all data columns")

    return data, flags


def _setup():
    # NOTE:
    # the import is needed to trigger the registration
    # of the built-in (test-)functions
    import saqc.funcs

    # warnings
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")


_setup()


class SaQC:
    def __init__(self, flagger, data, flags=None, nodata=np.nan, error_policy="raise"):
        data, flags = _prepInput(flagger, data, flags)
        self._data = data
        self._nodata = nodata
        self._flagger = self._initFlagger(data, flagger, flags)
        self._error_policy = error_policy
        # NOTE: will be filled by calls to `_wrap`
        self._to_call: List[Dict[str, Any]] = []  # todo fix the access everywhere

    def _initFlagger(self, data, flagger, flags):
        """ Init the internal flagger object.

        Ensures that all data columns are present and user passed flags from
        a flags frame and/or an already initialised flagger are used.
        If columns overlap the passed flagger object is prioritised.
        """
        # ensure all data columns
        merged = flagger.initFlags(data)
        if flags is not None:
            merged = merged.merge(flagger.initFlags(flags=flags))
        if flagger.initialized:
            merged = merged.merge(flagger)
        return merged

    def readConfig(self, fname):

        config = readConfig(fname)

        out = deepcopy(self)
        for func, field, kwargs, plot, lineno, expr in config:
            if isQuoted(field):
                kwargs["regex"] = True
                field = field[1:-1]
            kwargs["field"] = field
            kwargs["plot"] = plot
            out = out._wrap(func, lineno=lineno, expr=expr)(**kwargs)
        return out

    def evaluate(self):
        """
        Realize all the registered calculations and return a updated SaQC Object

        Paramters
        ---------

        Returns
        -------
        An updated SaQC Object incorporating the requested computations
        """

        # NOTE: It would be nicer to separate the plotting into an own
        #       method instead of intermingling it with the computation
        data, flagger = self._data, self._flagger

        for func_dump in self._to_call:
            func_name = func_dump['func_name']
            func_kws = func_dump['func_kws']
            field = func_dump['field']
            plot = func_dump["ctrl_kws"]["plot"]
            logger.debug(f"processing: {field}, {func_name}, {func_kws}")

            try:
                t0 = timeit.default_timer()
                data_result, flagger_result = _saqcCallFunc(func_dump, data, flagger)

            except Exception as e:
                t1 = timeit.default_timer()
                logger.debug(f"{func_name} failed after {t1 - t0} sec")
                _handleErrors(e, func_dump, self._error_policy)
                continue
            else:
                t1 = timeit.default_timer()
                logger.debug(f"{func_name} finished after {t1 - t0} sec")

            if plot:
                plotHook(
                    data_old=data,
                    data_new=data_result,
                    flagger_old=flagger,
                    flagger_new=flagger_result,
                    sources=[],
                    targets=[field],
                    plot_name=func_name,
                )

            data = data_result
            flagger = flagger_result

        if any([fdump["ctrl_kws"]["plot"] for fdump in self._to_call]):
            plotAllHook(data, flagger)

        return SaQC(flagger, data, nodata=self._nodata, error_policy=self._error_policy)

    def getResult(self):
        """
        Realized the registerd calculations and return the results

        Returns
        -------
        data, flagger: (DictOfSeries, DictOfSeries)
        """

        realization = self.evaluate()
        return realization._data, realization._flagger

    def _wrap(self, func_name, lineno=None, expr=None):
        def inner(field: str, *args, regex: bool = False, to_mask=None, plot=False, **kwargs):
            fields = [field] if not regex else self._data.columns[self._data.columns.str.match(field)]

            if func_name in ("flagGeneric", "procGeneric"):
                # NOTE:
                # We need to pass `nodata` to the generic functions
                # (to implement stuff like `ismissing`). As we
                # should not interfere with proper nodata attributes
                # of other test functions (e.g. `flagMissing`) we
                # special case the injection
                kwargs.setdefault('nodata', self._nodata)

            # to_mask is a control keyword
            ctrl_kws = {
                **(FUNC_MAP[func_name]["ctrl_kws"]),
                'to_mask': to_mask, "plot": plot,
                'lineno': lineno,
                'expr': expr
            }
            func = FUNC_MAP[func_name]["func"]

            func_dump = {
                "func_name": func_name,
                "func": func,
                "func_args": args,
                "func_kws": kwargs,
                "ctrl_kws": ctrl_kws,
            }

            out = deepcopy(self)
            for field in fields:
                dump_copy = {**func_dump, "field": field}
                out._to_call.append(dump_copy)
            return out

        return inner

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to
        __getattr__. We use this mechanism to make the
        registered functions as `SaQC`-methods without
        actually implementing them.
        """
        if key not in FUNC_MAP:
            raise AttributeError(f"no such attribute: '{key}'")
        return self._wrap(key)


def _saqcCallFunc(func_dump, data, flagger):
    func = func_dump['func']
    func_name = func_dump['func_name']
    func_args = func_dump['func_args']
    func_kws = func_dump['func_kws']
    field = func_dump['field']
    ctrl_kws = func_dump['ctrl_kws']
    to_mask = ctrl_kws['to_mask']

    # NOTE:
    # when assigning new variables to `data`, the respective
    # field is missing in `flags`, so we add it if necessary in
    # order to keep the columns from `data` and `flags` in sync
    if field not in flagger.getFlags():
        flagger = flagger.merge(flagger.initFlags(data=pd.Series(name=field)))

    to_mask = flagger.BAD if to_mask is None else to_mask
    data_in = _maskData(data, flagger, to_mask)
    data_result, flagger_result = func(data_in, field, flagger, *func_args, **func_kws)
    data_result = _unmaskData(data, flagger, data_result, flagger_result, to_mask)

    return data_result, flagger_result


def _maskData(data, flagger, to_mask):
    # TODO: this is heavily undertested
    mask = flagger.isFlagged(flag=to_mask, comparator='==')
    data = data.copy()
    for c in data.columns:
        col_mask = mask[c].values
        if np.any(col_mask):
            col_data = data[c].values.astype(np.float64)
            col_data[col_mask] = np.nan
            data[c] = col_data
    return data


def _unmaskData(data_old, flagger_old, data_new, flagger_new, to_mask):
    # TODO: this is heavily undertested
    mask_old = flagger_old.isFlagged(flag=to_mask, comparator="==")
    mask_new = flagger_new.isFlagged(flag=to_mask, comparator="==")

    for c, right in data_new.indexes.iteritems():
        if c not in mask_old:
            continue
        left = mask_old[c].index
        col = data_new[c]
        col_data = col.values
        col_index = col.index
        # NOTE: ignore columns with changed indices (assumption: harmonization)
        if left.equals(right):
            # NOTE: Don't overwrite data, that was masked, but is not considered
            # flagged anymore and also respect newly set data on masked locations.
            mask = mask_old[c].values & mask_new[c].values & data_new[c].isna().values
            if np.any(mask):
                col_data[mask] = data_old[c].values[mask]
        data_old[c] = pd.Series(data=col_data, index=col_index)
    return data_old


