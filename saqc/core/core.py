#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .config import Fields, Params
from ..funcs import flagDispatch
from ..dsl import parseFlag
from ..lib.plotting import plot


def flagWindow(flagger, flags, mask, direction='fw', window=0, **kwargs) -> pd.Series:
    fw = False
    bw = False
    f = flagger.isFlagged(flags) & mask

    if isinstance(window, int):
        x = f.rolling(window=window + 1).sum()
        if direction in ['fw', 'both']:
            fw = x.fillna(method='bfill').astype(bool)
        if direction in ['bw', 'both']:
            bw = x.shift(-window).fillna(method='bfill').astype(bool)
    else:
        # time-based windows
        if direction in ['bw', 'both']:
            raise NotImplementedError
        fw = f.rolling(window=window, closed='both').sum().astype(bool)

    fmask = bw | fw
    flags.loc[fmask] = flagger.setFlag(flags.loc[fmask], **kwargs)
    return flags


def flagPeriod(flagger, flags, mask=True, flag_period=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_period, **kwargs)


def flagNext(flagger, flags, mask=True, flag_values=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_values, **kwargs)


def assignTypeSafe(df, colname, rhs):
    """
    Works around a pandas issue: when assigning a
    data frame with differing columns dtypes,
    all columns are converted to the most generic
    of the dtypes
    """
    # do not use .loc here, as it fails silently :/
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
        varname, _, _, assign = configrow
        if varname not in flags and \
                (varname in data or varname not in data and assign is True):
            col_flags = flagger.initFlags(pd.DataFrame(index=data.index,
                                                       columns=[varname]))
            flags = col_flags if flags.empty else flags.join(col_flags)
    return flags


def runner(metafname, flagger, data, flags=None, nodata=np.nan):

    meta = prepareMeta(readMeta(metafname), data)
    # NOTE: split meta into the test and some 'meta' data
    fields = [Fields.VARNAME, Fields.START, Fields.END, Fields.ASSIGN]
    tests = meta[meta.columns.to_series().filter(regex=Fields.FLAGS)]
    meta = meta[fields]

    plotvars = []

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

        for idx, (varname, start_date, end_date, _) in meta.iterrows():

            flag_test = testcol[idx]
            if pd.isnull(flag_test):
                continue

            func_name, flag_params = parseFlag(flag_test)

            if varname not in data and varname not in flags:
                continue

            dchunk = data.loc[start_date:end_date]
            if dchunk.empty:
                continue

            fchunk = flags.loc[start_date:end_date]

            try:
                dchunk, ffchunk = flagDispatch(func_name,
                                               dchunk, fchunk.copy(),
                                               varname,
                                               flagger, nodata=nodata,
                                               **flag_params)
            except NameError:
                raise NameError(
                    f"function name {func_name} is not definied (variable '{varname}, 'line: {idx + 1})")

            old = flagger.getFlags(fchunk[varname])
            new = flagger.getFlags(ffchunk[varname])
            mask = old != new

            # flag a timespan after the condition is met
            if Params.FLAGPERIOD in flag_params:
                periodflags = flagPeriod(flagger, ffchunk[varname], mask, func_name=func_name, **flag_params)
                ffchunk = assignTypeSafe( ffchunk, varname, periodflags)

            # flag a certain amount of values after condition is met
            if Params.FLAGVALUES in flag_params:
                valueflags = flagNext(flagger, ffchunk[varname], mask, func_name=func_name, **flag_params)
                ffchunk = assignTypeSafe(ffchunk, varname, valueflags)

            if flag_params.get(Params.PLOT, False):
                plotvars.append(varname)
                mask = old != flagger.getFlags(ffchunk[varname])
                plot(dchunk, ffchunk, mask, varname, flagger, title=flag_test)

            data.loc[start_date:end_date] = dchunk
            flags.loc[start_date:end_date] = ffchunk.squeeze()

        flagger.nextTest()

    # plot all together
    if plotvars:
        plot(data, flags, True, plotvars, flagger)

    return data, flags


def readMeta(fname):
    return pd.read_csv(fname, delimiter=",", comment="#")


def prepareMeta(meta, data):
    # NOTE: an option needed to only pass tests within a file and deduce
    #       everything else from data

    # no dates given, fall back to the available index range
    if Fields.START not in meta:
        meta = meta.assign(**{Fields.START: np.nan})
    if Fields.END not in meta:
        meta = meta.assign(**{Fields.END: np.nan})

    meta = meta.fillna(
        {Fields.END: data.index.max(),
         Fields.START: data.index.min()})

    if Fields.ASSIGN not in meta:
        meta = meta.assign(**{Fields.ASSIGN: False})

    # rows without a variables name don't help much
    meta = meta.dropna(subset=[Fields.VARNAME])

    dtype = np.datetime64 if isinstance(data.index, pd.DatetimeIndex) else int

    meta[Fields.START] = meta[Fields.START].astype(dtype)
    meta[Fields.END] = meta[Fields.END].astype(dtype)

    return meta
