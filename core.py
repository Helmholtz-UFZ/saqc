#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil

import numpy as np
import pandas as pd

from config import Fields, FUNCMAP, NODATA
from funcs import Params
from dsl import evalCondition, parseFlag
from flagger import PositionalFlagger, BaseFlagger
from lib.types import ArrayLike


def _inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def _periodToTicks(period, freq):
    return int(ceil(pd.to_timedelta(period)/pd.to_timedelta(freq)))


def flagNext(flagger: BaseFlagger, flags: ArrayLike, n: int) -> ArrayLike:
    """
    to_flag: Union[np.ndarray[bool], pd.Series[bool]]
    """
    for name, values in flags.iteritems():
        idx = np.nonzero(flagger.isFlagged(values))[0]
        mask = flagger.isFlagged(values)
        for nn in range(1, n + 1):
            nn_idx = np.clip(idx + nn, a_min=None, a_max=len(values) - 1)
            nn_idx_unflagged = nn_idx[~flagger.isFlagged(values[nn_idx])]
            values[nn_idx_unflagged] = values[nn_idx_unflagged - nn]
    return flags



    return flags


def flaggingRunner(meta, flagger, data, flags=None):

    # TODO: if flags is not None, check its structure
    if flags is None:
        flags = flagger.emptyFlags(data)
    else:
        if not all(flags.columns == flagger.emptyFlags(data.iloc[0]).columns):
            raise TypeError("structure of given flag does not "
                            "correspond to flagger requirements")

    # NOTE:
    # we need an index frequency in order to calculate ticks
    # from given periods further down the road
    data.index.freq = _inferFrequency(data)
    assert data.index.freq, "no frequency deducable from timeseries"

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

        for _, configrow in meta.iterrows():

            flag_test = configrow[flag_field]
            if pd.isnull(flag_test):
                continue

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

            flag_params = parseFlag(flag_test)
            flag_name = flag_params[Params.NAME]

            # NOTE: higher flags might be overwritten by lower ones
            func = FUNCMAP.get(flag_name, None)
            if func:
                dchunk, fchunk = func(dchunk, fchunk, varname,
                                      flagger, nodata=NODATA, **flag_params)
            else:
                raise RuntimeError(
                    "malformed flag field ('{:}') for variable: {:}"
                    .format(flag_test, varname))


            # flag a timespan after the condition is met,
            # duration given in 'flag_period'
            flag_period = flag_params.pop(Params.FLAGPERIOD, None)
            if flag_period:
                flag_params[Params.FLAGVALUES] = _periodToTicks(flag_period,
                                                                data.index.freq)

            # flag a certain amount of values after condition is met,
            # number given in 'flag_values'
            flag_values = flag_params.pop(Params.FLAGVALUES, None)
            if flag_values:
                to_flag = flagNext(to_flag, flag_values)


            flagger.nextTest()
            data.loc[start_date:end_date] = dchunk
            flags.loc[start_date:end_date] = fchunk
    return data, flags


def prepareMeta(meta, data):
    # NOTE: an option needed to only pass test within an file and deduce
    #       everything else from data
    # no dates given, fall back to the available date range
    if Fields.STARTDATE not in meta:
        meta = meta.assign(**{Fields.STARTDATE: np.nan})
    if Fields.ENDDATE not in meta:
        meta = meta.assign(**{Fields.ENDDATE: np.nan})
    meta = meta.fillna(
        {Fields.ENDDATE: data.index.max(),
         Fields.STARTDATE: data.index.max()})
    meta = meta.dropna(subset=[Fields.VARNAME])
    meta[Fields.STARTDATE] = pd.to_datetime(meta[Fields.STARTDATE])
    meta[Fields.ENDDATE] = pd.to_datetime(meta[Fields.ENDDATE])
    return meta


def readData(fname):
    data = pd.read_csv(
        fname, index_col="Date Time", parse_dates=True,
        na_values=["-9999", "-9999.0"], low_memory=False)
    data.columns = [c.split(" ")[0] for c in data.columns]
    data = data.reindex(
        pd.date_range(data.index.min(), data.index.max(), freq="10min"))
    return data


if __name__ == "__main__":

    datafname = "resources/data.csv"
    metafname = "resources/meta.csv"
    data = readData(datafname)
    meta = prepareMeta(pd.read_csv(metafname), data)
    flagger = PositionalFlagger()
    pdata, pflags = flaggingRunner(meta, flagger, data)
