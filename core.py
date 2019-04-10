#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil

import numpy as np
import pandas as pd

from config import Fields, Params
from funcs import flagDispatch
from dsl import parseFlag
from flagger import PositionalFlagger, BaseFlagger


class FlagParams:
    FLAG = "flag"
    PERIODE = "flag_period"
    VALUES = "flag_values"
    ASSIGN = "assign_to"


def _inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def _periodToTicks(period, freq):
    return int(ceil(pd.to_timedelta(period)/pd.to_timedelta(freq)))


def flagNext(flagger: BaseFlagger, flags: pd.Series, n: int) -> pd.Series:
    idx = np.where(flagger.isFlagged(flags))[0]
    for nn in range(1, n + 1):
        nn_idx = np.clip(idx + nn, a_min=None, a_max=len(flags) - 1)
        nn_idx_unflagged = nn_idx[~flagger.isFlagged(flags.iloc[nn_idx])]
        flags.values[nn_idx_unflagged] = flags.iloc[nn_idx_unflagged - nn]
    return flags


def runner(meta, flagger, data, flags=None, nodata=np.nan):

    if flags is None:
        flags = flagger.emptyFlags(data)
    else:
        if not all(flags.columns == flagger.emptyFlags(data.iloc[0]).columns):
            raise TypeError("structure of given flag does not "
                            "correspond to flagger requirements")

    # NOTE:
    # We need an index frequency in order to calculate ticks
    # from given periods further down the road. Maybe this
    # restriction should only be enforced when we really
    # need a time series...
    data.index.freq = _inferFrequency(data)
    if not data.index.freq:
        raise TypeError("cannot infer time frequency from dataset")

    # the required meta data columns
    fields = [Fields.VARNAME, Fields.STARTDATE, Fields.ENDDATE]

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

            func_name, flag_params = parseFlag(flag_test)

            assign_to = flag_params.get(FlagParams.ASSIGN)
            if assign_to:
                dummy = pd.DataFrame(index=data.index, columns=[assign_to])
                flags = flags.join(flagger.emptyFlags(dummy))

            varname, start_date, end_date = configrow[fields]
            if varname not in data:
                continue

            dchunk = data.loc[start_date:end_date]
            if dchunk.empty:
                continue

            # NOTE:
            # within the activation period of a variable, the flag will
            # be initialized if necessary
            fchunk = (flags
                      .loc[start_date:end_date]
                      .fillna({varname: flagger.no_flag}))

            try:
                dchunk, fchunk = flagDispatch(func_name,
                                              dchunk, fchunk, varname,
                                              flagger, nodata=nodata,
                                              **flag_params)
            except NameError:
                raise NameError(
                    f"function name {func_name} is not definied (variable '{varname}, 'line: {idx + 1})")


            # flag a timespan after the condition is met,
            # duration given in 'flag_period'
            flag_period = flag_params.pop(Params.FLAGPERIOD, None)
            if flag_period:
                flag_params[Params.FLAGVALUES] = _periodToTicks(flag_period, data.index.freq)

            # flag a certain amount of values after condition is met,
            # number given in 'flag_values'
            flag_values = flag_params.pop(Params.FLAGVALUES, None)
            if flag_values:
                fchunk[varname] = flagNext(flagger, fchunk[varname], flag_values)

            data.loc[start_date:end_date] = dchunk
            flags.loc[start_date:end_date] = fchunk

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

    data = readData(datafname, index_col="Date Time", nans=["-9999", "-9999.0"])
    meta = prepareMeta(pd.read_csv(metafname), data)

    flagger = PositionalFlagger()
    pdata, pflags = runner(meta, flagger, data)
