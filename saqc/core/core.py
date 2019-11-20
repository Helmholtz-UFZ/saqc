#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .reader import readConfig, prepareConfig
from .config import Fields
from .evaluator import evalExpression
from ..lib.plotting import plot
from ..lib.tools import setup
from ..flagger import FlaggerTemplate, BaseFlagger, SimpleFlagger, DmpFlagger


def collectVariables(meta, data):
    """
    find every relevant variable
    """
    # NOTE: get to know every variable from meta
    flags = [] #data.columns.tolist()
    for idx, configrow in meta.iterrows():
        varname = configrow[Fields.VARNAME]
        assign = configrow[Fields.ASSIGN]
        if varname in data:
            flags.append(varname)
        elif varname not in flags and assign is True:
            flags.append(varname)
    return flags


def _check_input(data, flagger):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be of type pd.DataFrame')

    if not isinstance(flagger, FlaggerTemplate):
        flaggerlist = [BaseFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f'flagger must be of type {flaggerlist} or any inherit class from {FlaggerTemplate}')


def runner(metafname, flagger, data, flags=None, nodata=np.nan):

    setup()
    _check_input(data, flagger)

    # NOTE: prep the config data
    # TODO: add a checkConfig call
    meta = prepareConfig(readConfig(metafname), data)
    tests = meta[meta.columns.to_series().filter(regex=Fields.TESTS)]
    meta = meta[meta.columns.difference(tests.columns)]

    # NOTE: prep the flagger
    if flags is None:
        flag_cols = collectVariables(meta, data)
        flagger = flagger.initFlags(pd.DataFrame(index=data.index, columns=flag_cols))
    else:
        flagger = flagger.initFromFlags(flags)

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

            if varname not in data and varname not in flagger.getFlags().columns:
                continue

            dchunk = data.loc[start_date:end_date]
            if dchunk.empty:
                continue

            fchunk = flagger.getFlagger(loc=dchunk.index)

            dchunk, fchunk = evalExpression(
                flag_test,
                data=dchunk, field=varname,
                flagger=fchunk, nodata=nodata)

            data.loc[start_date:end_date] = dchunk
            # import pdb; pdb.set_trace()
            flagger.setFlags(field=varname, loc=dchunk.index, flag=fchunk.getFlags(field=varname))
            # flagger._flags.loc[start_date:end_date] == fchunk._flags

        # NOTE: this method should be removed
        flagger.nextTest()

    # plot all together
    plotvars = meta[meta[Fields.PLOT]][Fields.VARNAME].tolist()
    if plotvars:
        plot(data, flags, True, plotvars, flagger)

    return data, flagger #.getFlags()
