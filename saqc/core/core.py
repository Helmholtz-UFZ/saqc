#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODOS:
  - integrate plotting into the api
  - `data` and `flagger` as arguments to `getResult`
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Callable, Sequence
from dataclasses import dataclass, replace

import pandas as pd
import dios
import numpy as np
import timeit
import inspect

from saqc.lib.plotting import plotHook, plotAllHook
from saqc.flagger import BaseFlagger, CategoricalFlagger, SimpleFlagger, DmpFlagger
from saqc.core.register import FUNC_MAP
from saqc.funcs.tools import copy


logger = logging.getLogger("SaQC")


@dataclass
class FuncCtrl:
    "ctrl_kws"
    masking: str          # one of: "none", "field", "all"
    plot: bool
    lineno: Optional[int] = None
    expr: Optional[str] = None
    inplace: bool = False
    to_mask: Any = None   # flagger.FLAG constants or a list of those


@dataclass
class Func:
    name: str
    func: Callable[[pd.DataFrame, str, BaseFlagger, Any], Tuple[pd.DataFrame, BaseFlagger]]
    field: str
    kwargs: Dict[str, Any]
    ctrl: FuncCtrl
    regex: bool = False
    target: Optional[str] = None
    args: Tuple[Any] = tuple()


def _handleErrors(exc, func, policy):
    msg = f"Execution failed. Variable: '{func.field}', "
    if func.ctrl.lineno is not None and func.ctrl.expr is not None:
        msg += f"Config line {func.ctrl.lineno}: '{func.ctrl.expr}', "
    else:
        msg += f"Function: {func.name}(), parameters: '{func.kwargs}', "
    msg += f"Exception:\n{type(exc).__name__}: {exc}"

    if policy == "ignore":
        logger.debug(msg)
    elif policy == "warn":
        logger.warning(msg)
    else:
        logger.error(msg)
        raise exc


def _prepInput(flagger, data, flags):
    dios_like = (dios.DictOfSeries, pd.DataFrame)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if not isinstance(data, dios_like):
        raise TypeError("'data' must be of type pd.Series, pd.DataFrame or dios.DictOfSeries")

    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
            raise TypeError("'data' should not use MultiIndex")
        data = dios.to_dios(data)

    if not hasattr(data.columns, "str"):
        raise TypeError("expected dataframe columns of type string")

    if not isinstance(flagger, BaseFlagger):
        # NOTE: we should generate that list automatically,
        #       it won't ever be complete otherwise
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f"'flagger' must be of type {flaggerlist} or a subclass of {BaseFlagger}")

    if flags is not None:
        if not isinstance(flags, dios_like):
            raise TypeError("'flags' must be of type dios.DictOfSeries or pd.DataFrame")

        if isinstance(flags, pd.DataFrame):
            if isinstance(flags.index, pd.MultiIndex) or isinstance(flags.columns, pd.MultiIndex):
                raise TypeError("'flags' should not use MultiIndex")
            flags = dios.to_dios(flags)

        # NOTE: do not test all columns as they not necessarily need to be the same
        cols = flags.columns & data.columns
        if not (flags[cols].lengths == data[cols].lengths).all():
            raise ValueError("the length of 'flags' and 'data' need to be equal")

    if flagger.initialized:
        diff = data.columns.difference(flagger.getFlags().columns)
        if not diff.empty:
            raise ValueError("Missing columns in 'flagger': '{list(diff)}'")

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
    def __init__(self, flagger, data, flags=None, nodata=np.nan, to_mask=None, error_policy="raise"):
        data, flags = _prepInput(flagger, data, flags)
        self._data = data
        self._nodata = nodata
        self._to_mask = to_mask
        self._flagger = self._initFlagger(data, flagger, flags)
        self._error_policy = error_policy
        # NOTE: will be filled by calls to `_wrap`
        self._to_call: List[Func] = []  # todo fix the access everywhere

    def _initFlagger(self, data, flagger, flags):
        """ Init the internal flagger object.

        Ensures that all data columns are present and user passed flags from
        a flags frame and/or an already initialised flagger are used.
        If columns overlap the passed flagger object is prioritised.
        """
        # ensure all data columns
        merged = flagger.initFlags(data)
        if flags is not None:
            merged = merged.merge(flagger.initFlags(flags=flags), inplace=True)
        if flagger.initialized:
            merged = merged.merge(flagger, inplace=True)
        return merged

    def readConfig(self, fname):
        from saqc.core.reader import readConfig
        out = deepcopy(self)
        out._to_call.extend(readConfig(fname, self._flagger))
        return out

    def _expandFields(self, func, variables) -> Sequence[Func]:
        if not func.regex:
            return [func]

        out = []
        for field in variables[variables.str.match(func.field)]:
            out.append(replace(func, field=field))
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

        for func in self._to_call:
            for func in self._expandFields(func, data.columns.union(flagger._flags.columns)):
                logger.debug(f"processing: {func.field}, {func.name}, {func.kwargs}")

                try:
                    t0 = timeit.default_timer()
                    data_result, flagger_result = _saqcCallFunc(func, data, flagger)

                except Exception as e:
                    t1 = timeit.default_timer()
                    logger.debug(f"{func.name} failed after {t1 - t0} sec")
                    _handleErrors(e, func, self._error_policy)
                    continue
                else:
                    t1 = timeit.default_timer()
                    logger.debug(f"{func.name} finished after {t1 - t0} sec")

                if func.ctrl.plot:
                    plotHook(
                        data_old=data,
                        data_new=data_result,
                        flagger_old=flagger,
                        flagger_new=flagger_result,
                        sources=[],
                        targets=[func.field],
                        plot_name=func.name,
                    )

                data = data_result
                flagger = flagger_result

        if any([fdump.ctrl.plot for fdump in self._to_call]):
            plotAllHook(data, flagger)

        # This is much faster for big datasets that to throw everything in the constructor.
        # Simply because of _initFlagger -> merge() -> mergeDios() over all columns.
        new = SaQC(SimpleFlagger(), dios.DictOfSeries(), nodata=self._nodata, error_policy=self._error_policy)
        new._flagger, new._data = flagger, data
        return new

    def getResult(self, raw=False):
        """
        Realized the registered calculations and return the results

        Returns
        -------
        data, flagger: (DictOfSeries, DictOfSeries)
        """

        realization = self.evaluate()
        data, flagger = realization._data, realization._flagger
        if raw is False:
            return data.to_df(), flagger.toFrame()
        return data, flagger

    def _wrap(self, func_name):
        def inner(field: str, *args, target: str=None, regex: bool = False, to_mask=None, plot=False, inplace=False, **kwargs):

            kwargs.setdefault('nodata', self._nodata)

            func = FUNC_MAP[func_name]["func"]

            ctrl_kws = FuncCtrl(
                masking=FUNC_MAP[func_name]["masking"],
                to_mask=to_mask or self._to_mask,
                plot=plot,
                inplace=inplace,
                )

            func_dump = Func(
                name=func_name,
                func=func,
                field=field,
                target=target if target is not None else field,
                regex=regex,
                args=args,
                kwargs=kwargs,
                ctrl=ctrl_kws,
            )

            out = self if inplace else self.copy()
            out._to_call.append(func_dump)

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

    def copy(self):
        return deepcopy(self)


def _saqcCallFunc(func_dump, data, flagger):

    # NOTE:
    # We assure that all columns in data have an equivalent column in flags,
    # we might have more flagger columns though
    assert data.columns.difference(flagger.getFlags().columns).empty

    field = func_dump.field
    target = func_dump.target
    to_mask = func_dump.ctrl.to_mask
    masking = func_dump.ctrl.masking

    if (target != field) and (func_dump.regex is False):
        data, flagger = copy(data, field, flagger, target)
        field = target

    if masking == 'all':
        columns = data.columns
    elif masking == 'none':
        columns = []
    elif masking == 'field':
        columns = [field]
    else:
        raise ValueError(f"wrong use of `register(masking={masking})`")

    # warn if the user explicitly pass `to_mask=..` to a function that is
    # decorated by `register(masking='none')`, and so `to_mask` is ignored.
    if masking == 'none' and to_mask not in (None, []):
        logging.warning("`to_mask` is given, but the test ignore masking. Please refer to the documentation: TODO")
    to_mask = flagger.BAD if to_mask is None else to_mask

    data_in, mask = _maskData(data, flagger, columns, to_mask)
    data_result, flagger_result = func_dump.func(
        data_in, field, flagger,
        *func_dump.args, func_name=func_dump.name, **func_dump.kwargs)
    data_result = _unmaskData(data, mask, data_result, flagger_result, to_mask)

    # we check the passed function-kwargs after the actual call, because now "hard" errors would already have been
    # raised (Eg. `TypeError: got multiple values for argument 'data'`, when the user pass data=...)
    _warnForUnusedKwargs(func_dump, flagger)

    return data_result, flagger_result


def _maskData(data, flagger, columns, to_mask):
    # TODO: this is heavily undertested
    mask = flagger.isFlagged(field=columns, flag=to_mask, comparator='==')
    data = data.copy()
    for c in columns:
        col_mask = mask[c].values
        if np.any(col_mask):
            col_data = data[c].values.astype(np.float64)
            col_data[col_mask] = np.nan
            data[c] = col_data
    return data, mask


def _unmaskData(data_old, mask_old, data_new, flagger_new, to_mask):
    # TODO: this is heavily undertested

    # NOTE:
    # we only need to respect columns, that were masked,
    # and are also still present in new data.
    # this throws out:
    #  - any newly assigned columns
    #  - columns that were excluded from masking
    columns = mask_old.dropempty().columns.intersection(data_new.dropempty().columns)
    mask_new = flagger_new.isFlagged(field=columns, flag=to_mask, comparator="==")

    for col in columns:
        was_masked = mask_old[col]
        is_masked = mask_new[col]

        # if index changed we just go with the new data.
        # A test should use `register(masking='none')` if it changes
        # the index but, does not want to have all NaNs on flagged locations.
        if was_masked.index.equals(is_masked.index):
            mask = was_masked.values & is_masked.values & data_new[col].isna().values

            # reapplying old values on masked positions
            if np.any(mask):
                data = np.where(mask, data_old[col].values, data_new[col].values)
                data_new[col] = pd.Series(data=data, index=is_masked.index)

    return data_new


def _warnForUnusedKwargs(func_dump, flagger):
    """ Warn for unused kwargs, passed to a SaQC.function.

    Parameters
    ----------
    func_dump: dict
        Saqc internal data structure that hold all function info.
    flagger: saqc.flagger.BaseFlagger
        Flagger object.

    Returns
    -------
    None

    Notes
    -----
    A single warning via the logging module is thrown, if any number of
    missing kws are detected, naming each missing kw.
    """
    sig_kws = inspect.signature(func_dump.func).parameters

    # we need to ignore kwargs that are injected or
    # used to control the flagger
    ignore = flagger.signature + ('nodata',)

    missing = []
    for kw in func_dump.kwargs:
        # there is no need to check for
        # `kw in [KEYWORD_ONLY, VAR_KEYWORD or POSITIONAL_OR_KEYWORD]`
        # because this would have raised an error beforehand.
        if kw not in sig_kws and kw not in ignore:
            missing.append(kw)

    if missing:
        missing = ', '.join(missing)
        logging.warning(f"Unused argument(s): {missing}")


