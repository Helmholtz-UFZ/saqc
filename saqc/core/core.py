#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODOS:
  - integrate plotting into the api
  - `data` and `flagger` as arguments to `getResult`
"""

import logging
from copy import deepcopy
from operator import attrgetter
from typing import List, Tuple

import pandas as pd
import dios
import numpy as np

from saqc.lib.plotting import plotHook, plotAllHook
from saqc.lib.tools import isQuoted
from saqc.core.register import FUNC_MAP, SaQCFunc
from saqc.core.reader import readConfig
from saqc.flagger import BaseFlagger, CategoricalFlagger, SimpleFlagger, DmpFlagger


logger = logging.getLogger("SaQC")


def _handleErrors(exc, func, policy):
    msg = f"failed with:\n{type(exc).__name__}: {exc}"
    if func.lineno is not None and func.expr is not None:
        msg = f"config, line {func.lineno}: '{func.expr}' " + msg
    else:
        msg = f"function '{func.func}' with parameters '{func.kwargs}' " + msg

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
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f"flagger must be of type {flaggerlist} or any inherit class from {BaseFlagger}")

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

    return data, flags


def _setup(log_level):
    # NOTE:
    # the import is needed to trigger the registration
    # of the built-in (test-)functions
    import saqc.funcs

    # warnings
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")

    # logging
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SaQC:
    def __init__(self, flagger, data, flags=None, nodata=np.nan, log_level=logging.INFO, error_policy="raise"):
        _setup(log_level)
        data, flags = _prepInput(flagger, data, flags)
        self._flagger = flagger.initFlags(data)
        if flags is not None:
            self._flagger = self._flagger.merge(flagger.initFlags(flags=flags))
        self._data = data
        self._nodata = nodata
        self._error_policy = error_policy
        # NOTE: will be filled by calls to `_wrap`
        self._to_call: List[Tuple[str, SaQCFunc]] = []

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

    def getResult(self):
        data, flagger = self._data, self._flagger

        for field, func in self._to_call:

            try:
                data_result, flagger_result = func(data=data, flagger=flagger, field=field)
            except Exception as e:
                _handleErrors(e, func, self._error_policy)
                continue

            if func.plot:
                plotHook(
                    data_old=data, data_new=data_result,
                    flagger_old=flagger, flagger_new=flagger_result,
                    sources=[], targets=[field], plot_name=func.__name__,
                )

            data = data_result
            flagger = flagger_result

        if any([func.plot for _, func in self._to_call]):
            plotAllHook(data, flagger)

        return data, flagger

    def _wrap(self, func, lineno=None, expr=None):

        def inner(field: str, *args, regex: bool = False, **kwargs):

            fields = [field] if not regex else self._data.columns[self._data.columns.str.match(field)]

            if func.__name__ in ("flagGeneric", "procGeneric"):
                # NOTE:
                # We need to pass `nodata` to the generic functions
                # (to implement stuff like `ismissing`). As we
                # should not interfere with proper nodata attributes
                # of other test functions (e.g. `flagMissing`) we
                # special case the injection
                kwargs["nodata"] = kwargs.get("nodata", self._nodata)

            out = deepcopy(self)
            for field in fields:
                f = SaQCFunc(func, *args, lineno=lineno, expression=expr, **kwargs)
                out._to_call.append((field, f))
            return out

        return inner

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to
        __getattr__. We use this mechanism to make the
        `RegisterFunc`s appear as `SaQC`-methods with
        actually implementing them.
        """
        if key not in FUNC_MAP:
            raise AttributeError(f"no such attribute: '{key}'")
        return self._wrap(FUNC_MAP[key])
