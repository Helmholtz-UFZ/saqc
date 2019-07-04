#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..lib.tools import valueRange, slidingWindowIndices, inferFrequency, estimateSamplingRate
from ..dsl import evalExpression
from ..core.config import Params


def flagDispatch(func_name, *args, **kwargs):
    func_map = {
        "manflag": flagManual,
        "mad": flagMad,
        "constant": flagConstant,
        "range": flagRange,
        "generic": flagGeneric,
        "soilMoistureByFrost": flagSoilMoistureBySoilFrost,
        "soilMoistureByPrecipitation": flagSoilMoistureByPrecipitationEvents}

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


def flagMad(data, flags, field, flagger, length, z, freq=None, **kwargs):
    d = data[field].copy()
    freq = inferFrequency(d) if freq is None else freq
    if freq is None:
        raise ValueError("freqency cannot inferred, provide `freq` as a param to mad().")
    winsz = int(pd.to_timedelta(length) / freq)
    median = d.rolling(window=winsz, center=True, closed='both').median()
    diff = abs(d - median)
    mad = diff.rolling(window=winsz, center=True, closed='both').median()
    mask = (mad > 0) & (0.6745 * diff > z * mad)
    flags.loc[mask, field] = flagger.setFlag(flags.loc[mask, field], **kwargs)
    return data, flags


def flagSoilMoistureBySoilFrost(data, flags, field, flagger, soil_temp_reference, tolerated_deviation,
                                frost_level=0, **kwargs):
    """Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series.
    :param flags:                       A dataframe holding the flags/flag-entries of "data"
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object.
                                        like thingies that refer to the data(including datestrings).
    :param tolerated_deviation:         total seconds, denoting the maximal temporal deviation,
                                        the soil frost states timestamp is allowed to have, relative to the
                                        data point to-be-flagged.
    :param soil_temp_reference:         A STRING, denoting the fields name in data,
                                        that holds the data series of soil temperature values,
                                        the to-be-flagged values shall be checked against.
    :param frost_level:                 Value level, the flagger shall check against, when evaluating soil frost level.
    """

    # retrieve reference series
    refseries = data[soil_temp_reference]
    ref_flags = flags[soil_temp_reference]
    ref_use = flagger.isFlagged(ref_flags, flag=flagger.flags.min()) | \
              flagger.isFlagged(ref_flags, flag=flagger.flags.unflagged())
    # drop flagged values:
    refseries = refseries[ref_use.values]
    # drop nan values from reference series, since those are values you dont want to refer to.
    refseries = refseries.dropna()

    # wrap around df.index.get_loc method, to catch key error in case of empty tolerance window:
    def check_nearest_for_frost(ref_date, ref_series, tolerance, check_level):

        try:
            # if there is no reference value within tolerance margin, following line will raise key error and
            # trigger the exception
            ref_pos = ref_series.index.get_loc(ref_date, method='nearest', tolerance=tolerance)
        except KeyError:
            # since test is not applicable: make no change to flag state
            return False

        # if reference value index is available, return comparison result (to determine flag)
        return ref_series[ref_pos] <= check_level

    # make temporal frame holding date index, since df.apply cant access index
    temp_frame = pd.Series(data.index)
    # get flagging mask ("True" denotes "bad"="test succesfull")
    mask = temp_frame.apply(check_nearest_for_frost, args=(refseries,
                                                           tolerated_deviation, frost_level))
    # apply calculated flags
    flags.loc[mask.values, field] = flagger.setFlag(flags.loc[mask.values, field], **kwargs)

    return data, flags


def flagSoilMoistureByPrecipitationEvents(data, flags, field, flagger, prec_reference, sensor_meas_depth=0,
                                          sensor_accuracy=0, soil_porosity=0, **kwargs):
    """Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
    precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
    surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
    moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
    to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

    NOTE1: np.nan entries in the input precipitation series will be regarded as susipicious and the test will be
    omited for every 24h interval including a np.nan entrie in the original precipitation sampling rate.
    Only entry "0" will be regarded as denoting "No Rainfall".

    NOTE2: The function wont test any values that are flagged suspicious anyway - this may change in a future version.


    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision.
    :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    :param prec_reference:              Fieldname of the precipitation meassurements column in data.
    :param sensor_meas_depth:           Measurement depth of the soil moisture sensor, [m].
    :param sensor_accuracy:             Accuracy of the soil moisture sensor, [-].
    :param soil_porosity:               Porosity of moisture sensors surrounding soil, [-].
    """

    # retrieve input sampling rate (needed to translate ref and data rates into each other):
    input_rate = estimateSamplingRate(data.index)

    # retrieve data series input:
    dataseries = data[field]
    # "nan" suspicious values (neither "unflagged" nor "min-flagged")
    data_flags = flags[field]
    data_use = flagger.isFlagged(data_flags, flag=flagger.flags.min()) | \
               flagger.isFlagged(data_flags, flag=flagger.flags.unflagged())
    # drop suspicious values
    dataseries = dataseries[data_use.values]
    # additionally, drop the nan values that result from any preceeding upsampling of the
    # measurements:
    dataseries = dataseries.dropna()
    # eventually, after dropping all nans, there is nothing left:
    if dataseries.empty:
        return (data, flags)
    # estimate original data sampling frequencie (the original series sampling rate may not match data-input sample
    # rate):
    moist_rate = estimateSamplingRate(dataseries.index)
    # resample dataseries to its original sampling rate (now certain, to only get nans, indeed denoting "missing" data)
    dataseries = dataseries.resample(moist_rate).asfreq()

    # retrieve reference series input
    refseries = data[prec_reference]
    # "nan" suspicious values (neither "unflagged" nor "min-flagged")
    ref_flags = flags[prec_reference]
    ref_use = flagger.isFlagged(ref_flags, flag=flagger.flags.min()) | \
              flagger.isFlagged(ref_flags, flag=flagger.flags.unflagged())
    # drop suspicious values
    refseries = refseries[ref_use.values]
    # additionally, drop the nan values that result from any preceeding upsampling of the
    # measurements:
    refseries = refseries.dropna()
    # eventually after dropping all nans, there is nothing left:
    if refseries.empty:
        return (data,flags)
    prec_rate = estimateSamplingRate(refseries.index)
    refseries.resample(prec_rate).asfreq()

    # get 24 h prec. monitor (this makes last-24h-rainfall-evaluation independent from preceeding entries)
    prec_count = refseries.rolling(window='1D').apply(lambda x: x.sum(skipna=False), raw=False)
    # upsample with zeros to input data sampling rate:
    prec_count = prec_count.resample(input_rate).pad()

    # now we can: project precipitation onto dataseries sampling (and stack result to be able to apply df.rolling())
    eval_frame = pd.merge(dataseries, prec_count, how='left', left_index=True, right_index=True).stack().reset_index()

    # following reshaping operations make all columns available to a function applied on rolling windows (rolling will
    # only eat one column of a dataframe at a time and doesnt like multi indexes as well)
    ef = eval_frame[0]
    ef.index = eval_frame['level_0']

    # make raise and std. dev tester function (returns False for values that
    # should be flagged bad and True respectively. (must be this way, since np.nan gets casted to True)))
    def prec_test(x):
        x_moist = x[0::2]
        x_rain = x[1::2]
        if x_moist[-1] > x_moist[-2]:
            if (x_moist[-1] - x_moist[0]) > 2*x_moist.std():
                return ~(x_rain[-1] <= (sensor_meas_depth*soil_porosity*sensor_accuracy))
            else:
                return True
        else:
            return True

    # get valid moisture raises:
    # rolling.apply should only get active every second entrie of the stacked frame,
    # so periods per window have to be calculated,
    # (this gives sufficiant conditian since window size controlls daterange:)
    periods = 2*int(24*60*60/moist_rate.n)
    invalid_raises = ~ef.rolling(window='1D', closed='both', min_periods=periods)\
        .apply(prec_test, raw=False).astype(bool)
    # undo stacking heritage (only every second entrie actually is holding an information:
    invalid_raises = invalid_raises[1::2]
    # retrieve indices referring to values-to-be-flagged-bad
    invalid_indices = invalid_raises.index[invalid_raises]
    # set Flags
    flags.loc[invalid_indices, field] = flagger.setFlag(flags.loc[invalid_indices, field], **kwargs)
    return (data, flags)
