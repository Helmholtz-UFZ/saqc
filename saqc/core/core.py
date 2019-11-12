#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .config import Fields
from .evaluator import evalExpression
from ..lib.plotting import plot
from ..lib.tools import setup


def assignTypeSafe(df, colname, rhs):
    """
    Works around a pandas issue: when assigning a
    data frame with differing columns dtypes,
    all columns are converted to the most generic
    of the dtypes
    """
    # do not use .loc here, as it fails silently :/
    rhs = rhs[colname]
    df[colname] = rhs
    if isinstance(rhs, pd.Series):
        dtypes = rhs.dtypes
    else:
        dtypes = {(colname, c): rhs.dtypes[c] for c in rhs.columns}
    return df.astype(dtypes)


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


def runner(metafname, flagger, data, flags=None, nodata=np.nan):

    setup()
    meta = prepareMeta(readMeta(metafname), data)
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


def readMeta(fname):
    return pd.read_csv(fname, delimiter=",", comment="#")


def prepareMeta(meta, data):
    # NOTE: an option needed to only pass tests within a file and deduce
    #       everything else from data

    # no dates given, fall back to the available index range
    for field in [Fields.VARNAME, Fields.TESTS, Fields.START, Fields.END, Fields.ASSIGN, Fields.PLOT]:
        if field not in meta:
            meta = meta.assign(**{field: np.nan})

    meta = meta.fillna({
        Fields.VARNAME: np.nan,
        Fields.TESTS: np.nan,
        Fields.START: data.index.min(),
        Fields.END: data.index.max(),
        Fields.ASSIGN: False,
        Fields.PLOT: False,
    })

    if meta[Fields.VARNAME].isna().any():
        raise TypeError(f"columns {Fields.VARNAME} is needed")

    tests = meta.filter(regex=Fields.TESTS)
    if tests.isna().all(axis=1).any():
        raise TypeError("at least one test must be given")

    dtype = np.datetime64 if isinstance(data.index, pd.DatetimeIndex) else int

    meta[Fields.START] = meta[Fields.START].astype(dtype)
    meta[Fields.END] = meta[Fields.END].astype(dtype)

    return meta
