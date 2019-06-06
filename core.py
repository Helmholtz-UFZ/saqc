#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from config import Fields, Params
from funcs import flagDispatch
from dsl import parseFlag
from flagger import PositionalFlagger, BaseFlagger


def inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


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
    flags[fmask] = flagger.setFlag(flags[fmask], **kwargs)
    return flags


def flagPeriod(flagger, flags, mask=True, flag_period=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_period, **kwargs)


def flagNext(flagger, flags, mask=True, flag_values=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_values, **kwargs)


def runner(meta, flagger, data, flags=None, nodata=np.nan):
    if flags is None:
        flags = pd.DataFrame(index=data.index)

    # the required meta data columns
    fields = [Fields.VARNAME, Fields.STARTDATE, Fields.ENDDATE, Fields.ASSIGN]

    # NOTE:
    # get to know every variable from meta
    # should go into a separate function
    for idx, configrow in meta.iterrows():
        varname, _, _, assign = configrow[fields]
        if varname not in flags and \
                (varname in data or varname not in data and assign is True):
            col_flags = flagger.initFlags(pd.DataFrame(index=data.index,
                                                       columns=[varname]))
            flags = col_flags if flags.empty else flags.join(col_flags)

    # NOTE:
    # the outer loop runs over the flag tests, the inner one over the
    # variables. Switching the loop order would complicate the
    # reference to flags from other variables within the dataset
    flag_fields = meta.columns.to_series().filter(regex=Fields.FLAGS)
    for flag_pos, flag_field in enumerate(flag_fields):

        # NOTE: just an optimization
        if meta[flag_field].dropna().empty:
            continue

        for idx, configrow in meta.iterrows():

            flag_test = configrow[flag_field]
            if pd.isnull(flag_test):
                continue

            varname, start_date, end_date, _ = configrow[fields]
            func_name, flag_params = parseFlag(flag_test)

            if varname not in data and varname not in flags:
                continue

            dchunk = data.loc[start_date:end_date].copy()
            if dchunk.empty:
                continue

            fchunk = flags.loc[start_date:end_date].copy()

            try:
                dchunk, fchunk = flagDispatch(func_name,
                                              dchunk, fchunk, varname,
                                              flagger, nodata=nodata,
                                              **flag_params)
            except NameError:
                raise NameError(
                    f"function name {func_name} is not definied (variable '{varname}, 'line: {idx + 1})")

            old = flagger.getFlags(flags.loc[start_date:end_date, varname])
            new = flagger.getFlags(fchunk[varname])
            mask = old != new

            # flag a timespan after the condition is met
            if Params.FLAGPERIOD in flag_params:
                fchunk[varname] = flagPeriod(flagger, fchunk[varname], mask, **flag_params)

            # flag a certain amount of values after condition is met
            if Params.FLAGVALUES in flag_params:
                fchunk[varname] = flagNext(flagger, fchunk[varname], mask, **flag_params)

            if Params.FLAGPERIOD in flag_params or Params.FLAGVALUES in flag_params:
                # hack as assignment above don't preserve categorical type
                fchunk = fchunk.astype({
                    c: flagger.flags for c in fchunk.columns if flagger.flag_fields[0] in c})

            data.loc[start_date:end_date] = dchunk
            flags[start_date:end_date] = fchunk.squeeze()

        flagger.nextTest()
    return data, flags


def prepareMeta(meta, data):
    # NOTE: an option needed to only pass tests within an file and deduce
    #       everything else from data

    # no dates given, fall back to the available date range
    if Fields.STARTDATE not in meta:
        meta = meta.assign(**{Fields.STARTDATE: np.nan})
    if Fields.ENDDATE not in meta:
        meta = meta.assign(**{Fields.ENDDATE: np.nan})
    meta = meta.fillna(
        {Fields.ENDDATE: data.index.max(),
         Fields.STARTDATE: data.index.min()})

    if Fields.ASSIGN not in meta:
        meta = meta.assign(**{Fields.ASSIGN: False})

    # rows without a variables name don't help much
    meta = meta.dropna(subset=[Fields.VARNAME])

    meta[Fields.STARTDATE] = pd.to_datetime(meta[Fields.STARTDATE])
    meta[Fields.ENDDATE] = pd.to_datetime(meta[Fields.ENDDATE])
    return meta


def readData(fname, index_col, nans):
    data = pd.read_csv(
        fname, index_col=index_col, parse_dates=True,
        na_values=nans, low_memory=False)
    data.columns = [c.split(" ")[0] for c in data.columns]
    data = data.reindex(
        pd.date_range(data.index.min(), data.index.max(), freq="10min"))
    return data


if __name__ == "__main__":
    datafname = "resources/data.csv"
    metafname = "resources/meta.csv"

    data = readData(datafname,
                    index_col="Date Time",
                    nans=["-9999", "-9999.0"])
    meta = prepareMeta(pd.read_csv(metafname), data)

    flagger = PositionalFlagger()
    pdata, pflags = runner(meta, flagger, data)
