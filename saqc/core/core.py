#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
TODOS:
  - integrate plotting into the api
  - `data` and `flagger` as arguments to `getResult`
"""

import logging
import copy as stdcopy
from typing import List, Tuple, Sequence, Union
from typing_extensions import Literal

import pandas as pd
import dios
import numpy as np
import timeit
import inspect

from saqc.common import *
from saqc.flagger.flags import init_flags_like, Flagger
from saqc.core.lib import APIController, ColumnSelector
from saqc.core.register import FUNC_MAP, SaQCFunction
from saqc.core.modules import FuncModules
from saqc.funcs.tools import copy
from saqc.lib.plotting import plotHook, plotAllHook


logger = logging.getLogger("SaQC")


def _handleErrors(exc: Exception, field: str, control: APIController, func: SaQCFunction, policy: Literal["ignore", "warn", "raise"]):
    message = "\n".join(
        [
            f"Exception:\n{type(exc).__name__}: {exc}",
            f"field: {field}",
            f"{func.errorMessage()}",
            f"{control.errorMessage()}"
        ]
    )

    if policy == "ignore":
        logger.debug(message)
    elif policy == "warn":
        logger.warning(message)
    else:
        logger.error(message)
        raise exc


# todo: shouldt this go to Saqc.__init__ ?
def _prepInput(data, flags):
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

    if flags is not None:

        if isinstance(flags, pd.DataFrame):
            if isinstance(flags.index, pd.MultiIndex) or isinstance(flags.columns, pd.MultiIndex):
                raise TypeError("'flags' should not use MultiIndex")

        if isinstance(flags, (dios.DictOfSeries, pd.DataFrame, Flagger)):
            # NOTE: only test common columns, data as well as flags could
            # have more columns than the respective other.
            cols = flags.columns & data.columns
            for c in cols:
                if not flags[c].index.equals(data[c].index):
                    raise ValueError(f"the index of 'flags' and 'data' missmatch in column {c}")

        # this also ensures float dtype
        if not isinstance(flags, Flagger):
            flags = Flagger(flags, copy=True)

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


class SaQC(FuncModules):

    def __init__(self, data, flags=None, nodata=np.nan, to_mask=None, error_policy="raise"):
        super().__init__(self)
        data, flagger = _prepInput(data, flags)
        self._data = data
        self._nodata = nodata
        self._to_mask = to_mask
        self._flagger = self._initFlagger(data, flags)
        self._error_policy = error_policy
        # NOTE: will be filled by calls to `_wrap`
        self._to_call: List[Tuple[ColumnSelector, APIController, SaQCFunction]] = []

    def _initFlagger(self, data, flagger: Union[Flagger, None]):
        """ Init the internal flagger object.

        Ensures that all data columns are present and user passed flags from
        a flags frame or an already initialised flagger are used.
        """
        if flagger is None:
            return init_flags_like(data)

        for c in flagger.columns.union(data.columns):
            if c in flagger:
                continue
            if c in data:
                flagger[c] = pd.Series(UNFLAGGED, index=data[c].index, dtype=float)
        return flagger

    def readConfig(self, fname):
        from saqc.core.reader import readConfig
        out = stdcopy.deepcopy(self)
        out._to_call.extend(readConfig(fname, self._flagger))
        return out

    def _expandFields(self, selector: ColumnSelector, func: SaQCFunction, variables: pd.Index) -> Sequence[Tuple[ColumnSelector, SaQCFunction]]:
        if not selector.regex:
            return [(selector, func)]

        out = []
        for field in variables[variables.str.match(selector.field)]:
            out.append((ColumnSelector(field=field, target=selector.target, regex=selector.regex), func))
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

        for selector, control, function in self._to_call:
            for sel, func in self._expandFields(selector, function, data.columns.union(flagger._flags.columns)):
                logger.debug(f"processing: {sel.field}, {func.name}, {func.keywords}")

                t0 = timeit.default_timer()
                try:
                    data_result, flagger_result = _saqcCallFunc(sel, control, func, data, flagger)
                except Exception as e:
                    t1 = timeit.default_timer()
                    logger.debug(f"{func.name} failed after {t1 - t0} sec")
                    _handleErrors(e, sel.field, control, func, self._error_policy)
                    continue
                else:
                    t1 = timeit.default_timer()
                    logger.debug(f"{func.name} finished after {t1 - t0} sec")

                if control.plot:
                    plotHook(
                        data_old=data,
                        data_new=data_result,
                        flagger_old=flagger,
                        flagger_new=flagger_result,
                        sources=[],
                        targets=[sel.field],
                        plot_name=func.name,
                    )

                data = data_result
                flagger = flagger_result

        if any([control.plot for _, control, _ in self._to_call]):
            plotAllHook(data, flagger)

        # This is way faster for big datasets, than to throw everything in the constructor.
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

    def _wrap(self, func: SaQCFunction):
        def inner(field: str, *fargs, target: str=None, regex: bool=False, to_mask=None, plot: bool=False, inplace: bool=False, **fkwargs) -> SaQC:

            control = APIController(
                masking=func.masking,
                to_mask=self._to_mask if to_mask is None else to_mask,
                plot=plot,
            )

            locator = ColumnSelector(
                field=field,
                target=target if target is not None else field,
                regex=regex,
            )

            partial = func.bind(*fargs, **{"nodata": self._nodata, "func_name": func.name, **fkwargs})

            out = self if inplace else self.copy(deep=True)
            out._to_call.append((locator, control, partial))

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
        return self._wrap(FUNC_MAP[key])

    def copy(self, deep=True):
        if deep:
            return stdcopy.deepcopy(self)
        return stdcopy.copy(self)


def _saqcCallFunc(locator, controller, function, data, flagger):
    # NOTE:
    # We assure that all columns in data have an equivalent column in flags,
    # we might have more flagger columns though
    assert data.columns.difference(flagger.getFlags().columns).empty

    field = locator.field
    target = locator.target
    to_mask = controller.to_mask
    masking = controller.masking

    if (target != field) and (locator.regex is False):
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
    data_result, flagger_result = function(data_in, field, flagger)
    data_result = _unmaskData(data, mask, data_result, flagger_result, to_mask)

    # we check the passed function-kwargs after the actual call, because now "hard" errors would already have been
    # raised (Eg. `TypeError: got multiple values for argument 'data'`, when the user pass data=...)
    _warnForUnusedKwargs(function, flagger)

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


def _warnForUnusedKwargs(func, flagger):
    """ Warn for unused kwargs, passed to a SaQC.function.

    Parameters
    ----------
    func: SaqcFunction
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
    sig_kws = inspect.signature(func.func).parameters

    # we need to ignore kwargs that are injected or
    # used to control the flagger
    ignore = flagger.signature + ("nodata", "func_name")

    missing = []
    for kw in func.keywords:
        # there is no need to check for
        # `kw in [KEYWORD_ONLY, VAR_KEYWORD or POSITIONAL_OR_KEYWORD]`
        # because this would have raised an error beforehand.
        if kw not in sig_kws and kw not in ignore:
            missing.append(kw)

    if missing:
        missing = ', '.join(missing)
        logging.warning(f"Unused argument(s): {missing}")
