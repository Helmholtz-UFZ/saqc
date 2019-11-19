#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .reader import readConfig, prepareConfig
from .config import Fields
from .evaluator import evalExpression
from ..lib.plotting import plot
from ..lib.tools import setup
from ..flagger import FlaggerTemplate, CategoricalFlagger, SimpleFlagger, DmpFlagger


def collectVariables(meta, flagger, data, flags):
    """
    find every relevant variable and add a respective
    column to the flags dataframe
    """
    # NOTE: get to know every variable from meta
    for idx, configrow in meta.iterrows():
        varname = configrow[Fields.VARNAME]
        assign = configrow[Fields.ASSIGN]
        if varname not in flags and \
                (varname in data or varname not in data and assign is True):
            col_flags = flagger.initFlags(pd.DataFrame(index=data.index,
                                                       columns=[varname]))
            flags = col_flags if flags.empty else flags.join(col_flags)
    return flags


def _check_input(data, flags, flagger):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be of type pd.DataFrame')

    if not isinstance(flagger, FlaggerTemplate):
        flaggerlist = [CategoricalFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f'flagger must be of type {flaggerlist} or any inherit class from {FlaggerTemplate}')

    if flags is None:
        return

    if not isinstance(flags, pd.DataFrame):
        raise TypeError('flags must be of type pd.DataFrame')

    refflags = flagger.initFlags(data)
    if refflags.shape != flags.shape:
        raise ValueError('flags has not the same dimensions as flagger.initFlags() would return')


def runner(metafname, flagger, data, flags=None, nodata=np.nan):
    setup()
    _check_input(data, flags, flagger)
    meta = prepareConfig(readConfig(metafname), data)
    # NOTE: split meta into the test and some 'meta' data
    tests = meta[meta.columns.to_series().filter(regex=Fields.TESTS)]
    meta = meta[meta.columns.difference(tests.columns)]

    # NOTE: prep the flags
    if flags is None:
        flags = pd.DataFrame(index=data.index)
    flags = collectVariables(meta, flagger, data, flags)

    # NOTE:
    # the outer loop runs over the flag tests, the inner one over the
    # variables. Switching the loop order would complicate the
    # reference to flags from other variables within the dataset
    for _, testcol in tests.iteritems():

        # NOTE: just an optimization
        if testcol.dropna().empty:
            continue

        for idx, configrow in meta.iterrows():
            varname = configrow[Fields.VARNAME]
            start_date = configrow[Fields.START]
            end_date = configrow[Fields.END]

            flag_test = testcol[idx]
            if pd.isnull(flag_test):
                continue

            if varname not in data and varname not in flags:
                continue

            dchunk = data.loc[start_date:end_date]
            if dchunk.empty:
                continue

            fchunk = flags.loc[start_date:end_date]

            dchunk, fchunk = evalExpression(
                flag_test,
                data=dchunk, flags=fchunk.copy(), field=varname,
                flagger=flagger, nodata=nodata)

            data.loc[start_date:end_date] = dchunk
            flags.loc[start_date:end_date] = fchunk.squeeze()

        # NOTE: this method should be removed
        flagger.nextTest()

    # plot all together
    plotvars = meta[meta[Fields.PLOT]][Fields.VARNAME].tolist()
    if plotvars:
        plot(data, flags, True, plotvars, flagger)

    return data, flags
