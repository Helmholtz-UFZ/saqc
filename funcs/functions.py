#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from lib.tools import valueRange, slidingWindowIndices
from dsl import evalExpression
from config import Params


def flagDispatch(func_name, *args, **kwargs):
    func_map = {
        "manflag": flagManual,
        "mad": flagMad,
        "constant": flagConstant,
        "range": flagRange,
        "generic": flagGeneric}

    func = func_map.get(func_name, None)
    if func is not None:
        return func(*args, **kwargs)
    raise NameError(f"function name {func_name} is not definied")


def flagGeneric(data, flags, field, flagger, nodata=np.nan, **flag_params):

    expression = flag_params[Params.FUNC]
    result = evalExpression(expression, flagger,
                            data, flags, field,
                            nodata=nodata)

    result = result.squeeze()

    if np.isscalar(result):
        raise TypeError(f"expression '{expression}' does not return an array")

    if not np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"expression '{expression}' does not return a boolean array")

    fchunk = flagger.setFlag(flags=flags.loc[result, field], **flag_params)

    flags.loc[result, field] = fchunk

    return data, flags


def flagConstant(data, flags, field, flagger, eps,
                 length, thmin=None, **kwargs):

    datacol = data[field]
    flagcol = flags[field]

    length = ((pd.to_timedelta(length) - data.index.freq)
              .to_timedelta64()
              .astype(np.int64))

    values = (datacol
              .mask((datacol < thmin) | datacol.isnull())
              .values
              .astype(np.int64))

    dates = datacol.index.values.astype(np.int64)

    mask = np.isfinite(values)

    for start_idx, end_idx in slidingWindowIndices(datacol.index, length):
        mask_chunk = mask[start_idx:end_idx]
        values_chunk = values[start_idx:end_idx][mask_chunk]
        dates_chunk = dates[start_idx:end_idx][mask_chunk]

        # we might have removed dates from the start/end of the
        # chunk resulting in a period shorter than 'length'
        # print (start_idx, end_idx)
        if valueRange(dates_chunk) < length:
            continue
        if valueRange(values_chunk) < eps:
            flagcol[start_idx:end_idx] = flagger.setFlags(flagcol[start_idx:end_idx], **kwargs)

    data[field] = datacol
    flags[field] = flagcol
    return data, flags


def flagManual(data, flags, field, flagger, **kwargs):
    return data, flags


def flagRange(data, flags, field, flagger, min, max, **kwargs):

    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flags.loc[mask, field] = flagger.setFlag(flags.loc[mask, field], **kwargs)
    return data, flags


def flagMad(data, flags, field, flagger, length, z, deriv, **kwargs):

    def _flagMad(data: np.ndarray, z: int, deriv: int) -> np.ndarray:
        # NOTE: numpy is at least twice as fast as numba.jit(nopython)
        # median absolute deviation
        for i in range(deriv):
            data[i+1:] = np.diff(data[i:])
            data[i] = np.nan
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data-median))
        tresh = mad * (z/0.6745)
        with np.errstate(invalid="ignore"):
            return (data < (median - tresh)) | (data > (median + tresh))

    datacol = data[field]
    flagcol = flags[field]

    values = (datacol
              .mask(flagger.isFlagged(flagcol))
              .values)

    window = pd.to_timedelta(length) - pd.to_timedelta(data.index.freq)
    mask = np.zeros_like(values, dtype=bool)

    for start_idx, end_idx in slidingWindowIndices(datacol.index, window, "1D"):
        mad_flags = _flagMad(values[start_idx:end_idx], z, deriv)
        # reset the mask
        mask[:] = False
        mask[start_idx:end_idx] = mad_flags
        flagcol[mask] = flagger.setFlag(flagcol[mask], **kwargs)

    flags[field] = flagcol

    return data, flags


def flagSoilMoistureBySoilFrost(data, flags, field, flagger, soil_temp_reference, tolerated_deviation,
                                frost_level=0, **kwargs):
    """Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    :param data:                        The pandas dataframe holding the data-to-be flagged.
    :param flags:                       A dataframe holding the flags/flag-entries of "data"
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object.
                                        like thingies that refer to the data(including datestrings).
    :param tolerated_deviation:         An offset alias, denoting the maximal temporal deviation,
                                        the soil frost states timestamp is allowed to have, relative to the
                                        data point to-be-flagged.
    :param soil_temp_reference:         A STRING, denoting the fields name in data,
                                        that holds the data series of soil temperature values,
                                        the to-be-flagged values shall be checked against.
    :param frost_level:                 Value level, the flagger shall check against, when evaluating soil frost level.
    """

    # TODO: (To ASK):HOW TO FLAG nan values in input frame? general question: what should a test test?
    # TODO: -> nan values with reference values that show frost, are flagged bad, nan values with reference value nan
    # TODO: as well, are not flagged (test not applicable-> no flag)
    # TODO: puffer zone for intermediate/fluktuating frost state

    # retrieve data series input:
    dataseries = pd.Series(data[field].values, index=pd.to_datetime(data.index))


    # retrieve reference input:
    #if reference is a string, it refers to data field

    # if reference series is part of input data frame, evaluate input data flags:
    flag_mask = flagger.isFlagged(flags)[soil_temp_reference]
    # retrieve reference series
    refseries = pd.Series(data[soil_temp_reference].values, index=pd.to_datetime(data.index))
    # drop flagged values:
    refseries = refseries.loc[~np.array(flag_mask)]
    # make refseries index a datetime thingy
    refseries.index = pd.to_datetime(refseries.index)
    # drop nan values from reference series, since those are values you dont want to refer to.
    refseries = refseries.dropna()

    # wrap around df.index.get_loc method to catch key error in case of empty tolerance window:
    def check_nearest_for_frost(ref_date, ref_series, tolerance, check_level):
        try:
            # if there is no reference value within tolerance margin, following line will rise key error and
            # trigger the exception
            ref_pos = ref_series.index.get_loc(ref_date, method='nearest', tolerance=tolerance)
        except KeyError:
            # since test is not applicable: make no change to flag state
            return False

        # if reference value index is available, return comparison result (to determine flag)
        return ref_series[ref_pos] <= check_level

    # make temporal frame holding dateindex, since df.apply cant access index
    temp_frame = pd.Series(dataseries.index)
    # get flagging mask
    mask = temp_frame.apply(check_nearest_for_frost, args=(refseries,
                                                           tolerated_deviation, frost_level))
    # apply calculated flags
    flags.loc[mask.values, field] = flagger.setFlag(flags.loc[mask, field], **kwargs)

    return data, flags
