#! /usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import dios

from saqc.core.reader import readConfig, checkConfig
from saqc.core.config import Fields
from saqc.core.evaluator import evalExpression
from saqc.lib.plotting import plotHook, plotAllHook
from saqc.lib.types import DiosLikeT
from saqc.flagger import BaseFlagger, CategoricalFlagger, SimpleFlagger, DmpFlagger


logger = logging.getLogger("SaQC")


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


def _convertInput(data, flags):
    if isinstance(data, pd.DataFrame):
        data = dios.to_dios(data)
    if isinstance(flags, pd.DataFrame):
        flags = dios.to_dios(flags)


def _checkAndConvertInput(data, flags, flagger):
    dios_like = (dios.DictOfSeries, pd.DataFrame)

    if not isinstance(data, dios_like):
        raise TypeError("data must be of type dios.DictOfSeries or pd.DataFrame")

    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            raise TypeError("the index of data is not allowed to be a multiindex")
        if isinstance(data.columns, pd.MultiIndex):
            raise TypeError("the columns of data is not allowed to be a multiindex")
        data = dios.to_dios(data)

    if not isinstance(flagger, BaseFlagger):
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f"flagger must be of type {flaggerlist} or any inherit class from {BaseFlagger}")

    if flags is not None:

        if not isinstance(flags, dios_like):
            raise TypeError("flags must be of type dios.DictOfSeries or pd.DataFrame")

        if isinstance(flags, pd.DataFrame):
            if isinstance(flags.index, pd.MultiIndex):
                raise TypeError("the index of flags is not allowed to be a multiindex")
            if isinstance(flags.columns, pd.MultiIndex):
                raise TypeError("the columns of flags is not allowed to be a multiindex")
            flags = dios.to_dios(flags)

        # NOTE: do not test all columns as they not necessarily need to be the same
        cols = flags.columns & data.columns
        if not (flags[cols].lengths == data[cols].lengths).all():
            raise ValueError("the length of values in flags and data does not match.")

    return data, flags


def _handleErrors(exc, configrow, test, policy):
    line = configrow[Fields.LINENUMBER]
    msg = f"config, line {line}, test: '{test}' failed with:\n{type(exc).__name__}: {exc}"
    if policy == "ignore":
        logger.debug(msg)
    elif policy == "warn":
        logger.warning(msg)
    else:
        raise Exception(msg)


def _setup(loglevel):
    pd.set_option("mode.chained_assignment", "warn")
    np.seterr(invalid="ignore")

    # logging setting
    logger.setLevel(loglevel)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run(
    config_file: str,
    flagger: BaseFlagger,
    data: DiosLikeT,
    flags: DiosLikeT = None,
    nodata: float = np.nan,
    log_level: str = "INFO",
    error_policy: str = "raise",
) -> (dios.DictOfSeries, BaseFlagger):

    _setup(log_level)
    data, flags = _checkAndConvertInput(data, flags, flagger)
    config = readConfig(config_file, data)

    # split config into the test and some 'meta' data
    tests = config.filter(regex=Fields.TESTS)
    meta = config[config.columns.difference(tests.columns)]

    # prepapre the flags
    flag_cols = _collectVariables(meta, data)
    flagger = flagger.initFlags(dios.DictOfSeries(data=data, columns=flag_cols))
    if flags is not None:
        flagger = flagger.merge(flagger.initFlags(flags=flags))

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
            flagger_chunk = flagger.slice(loc=dtslice)

            try:
                # actually run the tests
                data_chunk_result, flagger_chunk_result = evalExpression(
                    func, data=data_chunk, field=varname, flagger=flagger_chunk, nodata=nodata,
                )
            except Exception as e:
                _handleErrors(e, configrow, func, error_policy)
                continue

            if configrow[Fields.PLOT]:
                try:
                    plotHook(
                        data_old=data_chunk, data_new=data_chunk_result,
                        flagger_old=flagger_chunk, flagger_new=flagger_chunk_result,
                        sources=[], targets=[varname], plot_name=func,
                    )
                except Exception:
                    logger.exception(f"Plotting failed. \n"
                                     f"  config line:  {configrow[Fields.LINENUMBER]}\n"
                                     f"  expression:   {func}\n"
                                     f"  variable(s):  {[varname]}.")

            # NOTE:
            # time slicing support is currently disabled
            # flagger = flagger.merge(flagger_chunk_result)
            # data = combineDataFrames(data, data_chunk_result)
            flagger = flagger_chunk_result
            data = data_chunk_result

    plotfields = config[Fields.VARNAME][config[Fields.PLOT]]
    if len(plotfields) > 0:
        try:
            # to only show variables that have set the plot-flag
            # use: plotAllHook(data, flagger, targets=plotfields)
            plotAllHook(data, flagger)
        except Exception:
            logger.exception(f"Final plotting failed.")

    return data, flagger
