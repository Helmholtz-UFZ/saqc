#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..lib.tools import (
    valueRange,
    slidingWindowIndices,
    inferFrequency,
    estimateSamplingRate,
    retrieveTrustworthyOriginal)
from ..dsl import evalExpression
from ..core.config import Params

# NOTE: will be filled by calls to register
FUNC_MAP = {}


def register(name):

    def outer(func):
        FUNC_MAP[name] = func
        # func.__name__ = name

        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return outer


def flagDispatch(func_name, *args, **kwargs):
    func = FUNC_MAP.get(func_name, None)
    if func is not None:
        return func(*args, **kwargs)
    raise NameError(f"function name {func_name} is not definied")


@register("generic")
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


@register("constant")
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


@register("range")
def flagRange(data, flags, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flags.loc[mask, field] = flagger.setFlag(flags.loc[mask, field], **kwargs)
    return data, flags


@register("mad")
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


@register("flagSoilMoistureByFrost")
def flagSoilMoistureBySoilFrost(data, flags, field, flagger, soil_temp_reference, tolerated_deviation='1h',
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

    # SaQC policy: Only data that has been flagged by at least one test is allowed to be referred to:
    if soil_temp_reference not in flags.columns:
        print('WARNING - flagSoilMoistureBySoilFrost - :'
              'The reference variable {} is either not part of the passed data frame, or the value is not registered to'
              ' the flags frame. To register it to the flags frame and thus, make it available for reference within '
              'tests, you need to run at least one single target test on it. The test will be skipped.'
              .format(soil_temp_reference))
        return data, flags

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


@register("soilMoistureByPrecipitation")
def flagSoilMoistureByPrecipitationEvents(data, flags, field, flagger, prec_reference, sensor_meas_depth=0,
                                          sensor_accuracy=0, soil_porosity=0, std_factor=2, **kwargs):
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
    :param std_factor:                  The value determines by which rule it is decided, weather a raise in soil
                                        moisture is significant enough to trigger the flag test or not:
                                        Significants is assumed, if the raise is  greater then "std_factor" multiplied
                                        with the last 24 hours standart deviation.
    """

    # SaQC policy: Only data that has been flagged by at least one test is allowed to be referred to:
    if soil_temp_reference not in flags.columns:
        print('WARNING - flagSoilMoistureByPrecipitationEvents - :'
              'The reference variable {} is either not part of the passed data frame, or the value is not registered to'
              ' the flags frame. To register it to the flags frame and thus, make it available for reference within '
              'tests, you need to run at least one single target test on it. The test will be skipped.'
              .format(prec_reference))
        return data, flags

    # retrieve input sampling rate (needed to translate ref and data rates into each other):
    input_rate = estimateSamplingRate(data.index)
    dataseries, moist_rate = retrieveTrustworthyOriginal(data[field], flags[field], flagger)
    refseries, ref_rate = retrieveTrustworthyOriginal(data[prec_reference], flags[prec_reference], flagger)
    # abort processing if any of the measurement series has no valid entries!
    if moist_rate is np.nan:
        return data, flags
    if ref_rate is np.nan:
        return data, flags

    # get 24 h prec. monitor (this makes last-24h-rainfall-evaluation independent from preceeding entries)
    prec_count = refseries.rolling(window='1D').apply(lambda x: x.sum(skipna=False), raw=False)
    # upsample with zeros to input data sampling rate (we want to project the daysums onto the dataseries grid to
    # prepare for use of rolling:):
    prec_count = prec_count.resample(input_rate).pad()

    # now we can: project precipitation onto dataseries sampling (and stack result to be able to apply df.rolling())
    eval_frame = pd.merge(dataseries, prec_count, how='left', left_index=True, right_index=True).stack(dropna=False).reset_index()

    # following reshaping operations make all columns available to a function applied on rolling windows (rolling will
    # only eat one column of a dataframe at a time and doesnt like multi indexes as well)
    ef = eval_frame[0]
    ef.index = eval_frame['level_0']

    # make raise and std. dev tester function (returns False for values that
    # should be flagged bad and True respectively. (must be this way, since np.nan gets casted to True)))
    def prec_test(x, std_fac=std_factor):
        x_moist = x[0::2]
        x_rain = x[1::2]
        if x_moist[-1] > x_moist[-2]:
            if (x_moist[-1] - x_moist[0]) > std_fac*x_moist.std():
                return ~(x_rain[-1] <= (sensor_meas_depth*soil_porosity*sensor_accuracy))
            else:
                return True
        else:
            return True

    # rolling.apply should only get active every second entrie of the stacked frame,
    # so periods per window have to be calculated,
    # (this gives sufficiant conditian since window size controlls daterange:)
    periods = 2*int(24*60*60/moist_rate.n)
    invalid_raises = ~ef.rolling(window='1D', closed='both', min_periods=periods)\
        .apply(prec_test, raw=False).astype(bool)
    # undo stacking (only every second entrie actually is holding an information:
    invalid_raises = invalid_raises[1::2]
    # retrieve indices referring to values-to-be-flagged-bad
    invalid_indices = invalid_raises.index[invalid_raises]
    # set Flags
    flags.loc[invalid_indices, field] = flagger.setFlag(flags.loc[invalid_indices, field], **kwargs)
    return data, flags


@register("soilMoistureBySpikeDetection")
def flagSoilMoistureBySpikeDetection(data, flags, field, flagger, filter_window_size='3h',
                                     normalized_data=True, raise_factor=0.15, dev_cont_factor=0.2, noise_barrier=1,
                                     noise_window_size='12h', **kwargs):
    """Function detects and flags spikes in soil moisture data.
    A datapoint is considered a spike, if:
    (1) the quotient to its preceeding datapoint exceeds a certain bound
    (controlled by param "raise_factor")
    (2) the quotient of the datas second derivate at the preceeding and subsequent timestamps is close enough to 1.
    (controlled by param "dev_cont_factor")
    (3) the surrounding data is not too noisy. (Coefficient of Variation[+/- 12 h] < 1)
    (controlled by param "noise_barrier")

    Some things you should be conscious about when applying the test:

       NOTE1: You should run less complex tests, especially range-tests, the flag-by-precipitation-test and the
       flag-by-frost test previously to this one, since the spike check for any potential, unflagged spike,
       is relatively costly (1 x smoothing + 2 x deviating + 2 x condition application).

       NOTE2: Test will only provide meaningful results, if dataseries input of soilmoisture data is projected onto
       [0,1] interval, since, otherwise, the (coefficient of variation < barrier) condition is very likely always true.
       (Set normalized_data parameter to "False", to trigger automatic projection)



       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision.
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
       :param filter_window_size:          Offset string. For computing second derivate, a Savitzky-Golay filter
                                           is applied onto the timeseries. This Offset string controlls the sice of the
                                           smoothing window used.
       :param normalized_data:             Boolean. If False, the function projects the data-to-be-flagged onto the
                                           [0,1] interval, before testing for spikes.
       :param raise_factor:                A float, determinating the bound, the quotient of two consecutive values
                                           has to exceed, to be regarded as potentially spike. A value of 0.x will
                                           trigger the spike test for value y_t, if:
                                           (y_t)/(y_t-1) > 1 + 0.x or:
                                           (y_t)/(y_t-1) < 1 - 0.x.
       :param dev_cont_factor:             A float, determining the interval, the quotient of the datas second derivate
                                           around a potential spike has to be part of, to trigger spike flagging for
                                           this value. A datapoint y_t will pass this spike condition if,
                                           for dev_cont_factor = 0.x, and the second derivative y'' of y, the condition:
                                           1 - 0.x < abs((y''_t-1)/(y''_t+1)) < 1 + 0.x
                                           holds
       :param noise_barrier:               A float, determining the bound, the data noisy-ness around a potential spike
                                           should not exceed, in order to guarantee a justifyed judgement:
                                           Therefor the coefficient of variation (COVA) of all values in a certain window
                                           around the datapoint (controlled by param noise_window,
                                           but excluding the point itself, is evaluated and tested
                                           for: COVA < noise_barrier.
       :param noise_window_size:           Offset string, determining the size of the window, the coefficient of variation
                                           is calculated of, to determine data noisy-ness around a potential spike.
                                           The potential spike y_t will be centered in a window of expansion:
                                           [y_t - noise_window_size, y_t + noise_window_size].
    """

    # retrieve data series input at its original sampling rate
    # (Note: case distinction for pure series input to avoid error resulting from trying to access pd.Series[field]
    if isinstance(data, pd.Series):
        dataseries, data_rate = retrieveTrustworthyOriginal(data, flags, flagger)
    else:
        dataseries, data_rate = retrieveTrustworthyOriginal(data[field], flags[field], flagger)

    # abort processing if original series has no valid entries!
    if data_rate is np.nan:
        return data, flags

    # normalize data if nessecary
    if ~normalized_data:
        dataseries=dataseries/100

    quotient_series = dataseries/dataseries.shift(+1)
    spikes = (quotient_series > (1 + raise_factor)) | (quotient_series < (1 - raise_factor))
    spikes = spikes[spikes == True]
    # loop through spikes: (loop may sound ugly - but since the number of spikes is supposed to not exceed the
    # thousands for year data - a loop going through all the spikes instances is much faster than
    # a rolling window, rolling all through a stacked year dataframe )
    # calculate some values, repeatedly needed in the course of the loop:
    filter_window_seconds = pd.Timedelta.total_seconds(pd.Timedelta(filter_window_size))
    smoothing_periods = int(np.ceil((filter_window_seconds / data_rate.n)))
    lower_dev_bound = 1 - dev_cont_factor
    upper_dev_bound = 1 + dev_cont_factor

    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for spike in spikes.index:
        start_slice = spike - pd.Timedelta(filter_window_size)
        end_slice = spike + pd.Timedelta(filter_window_size)
        scnd_derivate = savgol_filter(dataseries[start_slice:end_slice],
                               window_length=smoothing_periods,
                               polyorder=2,
                               deriv=2)
        length = scnd_derivate.size
        test_ratio_1 = np.abs(scnd_derivate[int((length-1) / 2)] / scnd_derivate[int((length+1) / 2)])

        if lower_dev_bound < test_ratio_1 < upper_dev_bound:
            start_slice = spike - pd.Timedelta(noise_window_size)
            end_slice = spike + pd.Timedelta(noise_window_size)
            test_slice = dataseries[start_slice:end_slice].drop(spike)
            test_ratio_2 = np.abs(test_slice.var() / test_slice.mean())

            if test_ratio_2 > noise_barrier:
                spikes[spike] = False

        else:
            spikes[spike] = False

    spikes = spikes[spikes == True]
    flags.loc[spikes.index, field] = flagger.setFlag(flags.loc[spikes.index, field], **kwargs)
    return data, flags


@register("soilMoistureByBreakDetection")
def flagSoilMoistureByBreakDetection(data, flags, field, flagger, filter_window_size='3h', rel_change_rate_min=0.1,
                                     abs_change=0.01, first_der_factor=10, scnd_der_ratio_margin_1=0.05,
                                     scnd_der_ratio_margin_2=10, **kwargs):
    """Function flags breaks (jumps/drops) in soil moisture measurement series by. A measurement y_t is flagged a
    break, if:

    (1) y_t is changing relatively to its preceeding value by at least (100*rel_change_rate_min) percent
    (2) y_(t-1) is difffering from its preceeding value, by a margin of at least "abs_change"
    (3) Absolute second derivative |(y_t)'| has to be at least "first_der_factor" times as big as the arithmetic middle
        over all the first derivative values within a 24h window centered at t.
    (4) The ratio of the second derivatives at t and t+1 has to be "aproximately" 1.
        ([1-scnd__der_ration_margin_1, 1+scnd_ratio_margin_1])
    (5) The ratio of the second derivatives at t+1 and t+2 has to be larger than scnd_der_ratio_margin_2

    Note: As no reliable statement about the plausibility of the meassurements before and after the jump is possible,
    only the jump itself is flagged.



       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision (skips allowed).
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
       :param rel_change_rate_min          Float in [0,1]. See (1) of function descritpion above to learn more
       :param abs_change                   Float > 0. See (2) of function descritpion above to learn more.
       .param first_der_factor             Float > 0. See (3) of function descritpion above to learn more.
       .param scnd_der_ratio_margin_1      Float in [0,1]. See (4) of function descritpion above to learn more.
       .param scnd_der_ratio_margin_2      Float in [0,1]. See (5) of function descritpion above to learn more.
    """
    # retrieve data series input at its original sampling rate
    # (Note: case distinction for pure series input to avoid error resulting from trying to access pd.Series[field]
    if isinstance(data, pd.Series):
        dataseries, data_rate = retrieveTrustworthyOriginal(data, flags, flagger)
    else:
        dataseries, data_rate = retrieveTrustworthyOriginal(data[field], flags[field], flagger)
    # abort processing if original series has no valid entries!
    if data_rate is np.nan:
        return data, flags

    # relative - change - break criteria testing:
    abs_change = np.abs(dataseries.shift(+1) - dataseries)
    breaks = (abs_change > 0.01) & (abs_change / dataseries > 0.1)
    breaks = breaks[breaks == True]

    # First derivative criterion
    filter_window_seconds = pd.Timedelta.total_seconds(pd.Timedelta(filter_window_size))
    smoothing_periods = int(np.ceil((filter_window_seconds / data_rate.n)))
    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for brake in breaks.index:
        # slice out slice-to-be-filtered (with some safety extension of 3 hours)
        slice_start = brake - pd.Timedelta('12h') -pd.Timedelta('3h')
        slice_end = brake + pd.Timedelta('12h') + pd.Timedelta('3h')
        data_slice = dataseries[slice_start:slice_end]
        first_deri_series = pd.Series(data=savgol_filter(data_slice,
                                                         window_length=smoothing_periods,
                                                         polyorder=2,
                                                         deriv=1),
                                      index=data_slice.index)
        # condition constructing and testing:
        test_sum = abs((first_deri_series[brake - pd.Timedelta('12h'):brake + pd.Timedelta('12h')].sum()*10)
                       / first_deri_series.size)

        if abs(first_deri_series[brake]) > test_sum:
            # second derivative criterion:
            slice_start = brake - pd.Timedelta('3h')
            slice_end = brake + pd.Timedelta('3h')
            data_slice = data_slice[slice_start:slice_end]
            second_deri_series = pd.Series(data=savgol_filter(data_slice,
                                                             window_length=smoothing_periods,
                                                             polyorder=2,
                                                             deriv=2),
                                          index=data_slice.index)
            # criterion evaluation:
            first_second = 0.95 < abs((second_deri_series[brake] / second_deri_series.shift(-1)[brake])) < 1.05
            second_second = second_deri_series.shift(-1)[brake] / second_deri_series.shift(-2)[brake] > 10
            if (~ first_second) | (~ second_second):
                breaks[brake] = False
        else:
            breaks[brake] = False

    breaks = breaks[breaks == True]
    flags.loc[breaks.index, field] = flagger.setFlag(flags.loc[breaks.index, field], **kwargs)
    return data, flags


