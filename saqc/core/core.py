#! /usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import dios.dios as dios

from saqc.core.reader import readConfig, checkConfig
from saqc.core.config import Fields
from saqc.core.evaluator import evalExpression
from saqc.lib.plotting import plotHook, plotAllHook
from saqc.lib.tools import combineDataFrames
from saqc.flagger import BaseFlagger, CategoricalFlagger, SimpleFlagger, DmpFlagger


def _collectVariables(meta, data):
    """
    find every relevant variable
    """
    # NOTE: get to know every variable from meta
    variables = list(data.columns)
    for idx, configrow in meta.iterrows():
        varname = configrow[Fields.VARNAME]
        # assign = configrow[Fields.ASSIGN]
        if varname in variables:
            continue
        # if (varname in data):  # or (varname not in variables and assign is True):
        variables.append(varname)
    return variables


def _checkInput(data, flags, flagger):
    # fixme: also allow dataframe
    if not isinstance(data, dios.DictOfSeries):
        raise TypeError("data must be of type dios.DictOfSeries")

    # if isinstance(data.index, pd.MultiIndex):
    #     raise TypeError("the index of data is not allowed to be a multiindex")

    # if isinstance(data.columns, pd.MultiIndex):
    #     raise TypeError("the columns of data is not allowed to be a multiindex")

    if not isinstance(flagger, BaseFlagger):
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f"flagger must be of type {flaggerlist} or any inherit class from {BaseFlagger}")

    if flags is None:
        return

    if not isinstance(flags, dios.DictOfSeries):
        raise TypeError("flags must be of type dios.DictOfSeries")

    # if isinstance(data.index, pd.MultiIndex):
    #     raise TypeError("the index of data is not allowed to be a multiindex")

    # fixme: iter over common columns and check len
    # if len(data) != len(flags):
    #     raise ValueError("the index of flags and data has not the same length")

    # NOTE: do not test columns as they not necessarily must be the same


def _handleErrors(exc, configrow, test, policy):
    line = configrow[Fields.LINENUMBER]
    msg = f"config, line {line}, test: '{test}' failed with:\n{type(exc).__name__}: {exc}"
    if policy == "ignore":
        logging.debug(msg)
    elif policy == "warn":
        logging.warning(msg)
    else:
        raise Exception(msg)


def _setup():
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")


def run(
    config_file: str,
    flagger: BaseFlagger,
    data: dios.DictOfSeries,
    flags: dios.DictOfSeries = None,
    nodata: float = np.nan,
    error_policy: str = "raise",
) -> (dios.DictOfSeries, BaseFlagger):

    _setup()
    _checkInput(data, flags, flagger)
    config = readConfig(config_file, data)

    # split config into the test and some 'meta' data
    tests = config.filter(regex=Fields.TESTS)
    meta = config[config.columns.difference(tests.columns)]

    # prepapre the flags
    flag_cols = _collectVariables(meta, data)
    flagger = flagger.initFlags(dios.DictOfSeries(data=data, columns=flag_cols))
    if flags is not None:
        flagger = flagger.setFlagger(flagger.initFlags(flags=flags))

    # NOTE:
    # this checks comes late, but the compilation of
    # user-test needs fully prepared flags
    checkConfig(config, data, flagger, nodata)

    # NOTE:
    # the outer loop runs over the flag tests, the inner one over the
    # variables. Switching the loop order would complicate the
    # reference to flags from other variables within the dataset
    for _, testcol in tests.iteritems():

        # NOTE: just an optimization
        if testcol.dropna().empty:
            continue

        for idx, configrow in meta.iterrows():

            # store config params in some handy variables
            varname = configrow[Fields.VARNAME]
            start_date = configrow[Fields.START]
            end_date = configrow[Fields.END]

            func = testcol[idx]
            if pd.isnull(func):
                continue

            if varname not in data and varname not in flagger.getFlags():
                continue

            # NOTE:
            # time slicing support is currently disabled
            # prepare the data for the tests
            # dtslice = slice(start_date, end_date)
            dtslice = slice(None)
            data_chunk = data.loc[dtslice]
            if data_chunk.empty:
                continue
            flagger_chunk = flagger.getFlagger(loc=dtslice)

            try:
                # actually run the tests
                data_chunk_result, flagger_chunk_result = evalExpression(
                    func, data=data_chunk, field=varname, flagger=flagger_chunk, nodata=nodata,
                )
            except Exception as e:
                _handleErrors(e, configrow, func, error_policy)
                continue

            if configrow[Fields.PLOT]:
                plotHook(
                    data_chunk, data_chunk_result,
                    flagger_chunk, flagger_chunk_result,
                    [varname], plot_name=func,
                )

            # NOTE:
            # time slicing support is currently disabled
            # flagger = flagger.setFlagger(flagger_chunk_result)
            # data = combineDataFrames(data, data_chunk_result)
            flagger = flagger_chunk_result
            data = data_chunk_result

    # plotAllHook(data, flagger)

    return data, flagger
