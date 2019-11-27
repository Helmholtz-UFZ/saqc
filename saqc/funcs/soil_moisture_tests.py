#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from saqc.funcs.break_detection import flagBreaks_spektrumBased
from saqc.funcs.spike_detection import flagSpikes_spektrumBased
from saqc.funcs.register import register
from saqc.lib.tools import (
    estimateSamplingRate,
    retrieveTrustworthyOriginal,
    offset2periods,
)


@register("soilMoisture_spikes")
def flagSoilMoistureSpikes(
    data,
    field,
    flagger,
    raise_factor=0.15,
    dev_cont_factor=0.2,
    noise_barrier=1,
    noise_window_size="12h",
    noise_statistic="CoVar",
    filter_window_size=None,
    **kwargs
):

    """
    The Function provides just a call to flagSpikes_spektrumBased, with parameter defaults, that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.
    """

    return flagSpikes_spektrumBased(
        data,
        field,
        flagger,
        filter_window_size=filter_window_size,
        raise_factor=raise_factor,
        dev_cont_factor=dev_cont_factor,
        noise_barrier=noise_barrier,
        noise_window_range=noise_window_size,
        noise_statistic=noise_statistic,
        **kwargs
    )


@register("soilMoisture_breaks")
def flagSoilMoistureBreaks(
    data,
    field,
    flagger,
    diff_method="raw",
    filter_window_size="3h",
    rel_change_rate_min=0.1,
    abs_change_min=0.01,
    first_der_factor=10,
    first_der_window_size="12h",
    scnd_der_ratio_margin_1=0.05,
    scnd_der_ratio_margin_2=10,
    smooth_poly_order=2,
    **kwargs
):

    """
    The Function provides just a call to flagBreaks_spektrumBased, with parameter defaults that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    """
    return flagBreaks_spektrumBased(
        data,
        field,
        flagger,
        diff_method=diff_method,
        filter_window_size=filter_window_size,
        rel_change_min=rel_change_rate_min,
        abs_change_min=abs_change_min,
        first_der_factor=first_der_factor,
        first_der_window_range=first_der_window_size,
        scnd_der_ratio_margin_1=scnd_der_ratio_margin_1,
        scnd_der_ratio_margin_2=scnd_der_ratio_margin_2,
        smooth_poly_order=smooth_poly_order,
        **kwargs
    )


@register("soilMoisture_frost")
def flagSoilMoistureBySoilFrost(
    data,
    field,
    flagger,
    soil_temp_reference,
    tolerated_deviation="1h",
    frost_level=0,
    **kwargs
):

    """This Function is an implementation of the soil temperature based Soil Moisture flagging, as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    All parameters default to the values, suggested in this publication.

    Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series.
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object.
                                        like thingies that refer to the data(including datestrings).
    :param tolerated_deviation:         Offset String. Denoting the maximal temporal deviation,
                                        the soil frost states timestamp is allowed to have, relative to the
                                        data point to-be-flagged.
    :param soil_temp_reference:         A STRING, denoting the fields name in data,
                                        that holds the data series of soil temperature values,
                                        the to-be-flagged values shall be checked against.
    :param frost_level:                 Value level, the flagger shall check against, when evaluating soil frost level.
    """

    # retrieve reference series
    refseries = data[soil_temp_reference]
    ref_use = flagger.isFlagged(
        soil_temp_reference, flag=flagger.GOOD, comparator="=="
    ) | flagger.isFlagged(soil_temp_reference, flag=flagger.UNFLAGGED, comparator="==")
    # drop flagged values:
    refseries = refseries[ref_use.values]
    # drop nan values from reference series, since those are values you dont want to refer to.
    refseries = refseries.dropna()

    # skip further processing if reference series is empty:
    if refseries.empty:
        return data, flagger

    # wrap around df.index.get_loc method, to catch key error in case of empty tolerance window:
    def _checkNearestForFrost(ref_date, ref_series, tolerance, check_level):

        try:
            # if there is no reference value within tolerance margin, following line will raise key error and
            # trigger the exception
            ref_pos = ref_series.index.get_loc(
                ref_date, method="nearest", tolerance=tolerance
            )
        except KeyError:
            # since test is not applicable: make no change to flag state
            return False

        # if reference value index is available, return comparison result (to determine flag)
        return ref_series[ref_pos] <= check_level

    # make temporal frame holding date index, since df.apply cant access index
    temp_frame = pd.Series(data.index)
    # get flagging mask ("True" denotes "bad"="test succesfull")
    mask = temp_frame.apply(
        _checkNearestForFrost, args=(refseries, tolerated_deviation, frost_level)
    )
    # apply calculated flags
    flagger = flagger.setFlags(field, mask.values, **kwargs)
    return data, flagger


@register("soilMoisture_precipitation")
def flagSoilMoistureByPrecipitationEvents(
    data,
    field,
    flagger,
    prec_reference,
    sensor_meas_depth=0,
    sensor_accuracy=0,
    soil_porosity=0,
    std_factor=2,
    std_factor_range="24h",
    raise_reference=None,
    **kwargs
):

    """This Function is an implementation of the precipitation based Soil Moisture flagging, as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    All parameters default to the values, suggested in this publication. (excluding porosity,sensor accuracy and
    sensor depth)


    Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
    precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
    surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
    moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
    to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

    A data point y_t is flagged an invalid soil moisture raise, if:

    (1) y_t > y_(t-raise_reference)
    (2) y_t - y_(t-"std_factor_range") > "std_factor" * std(y_(t-"std_factor_range"),...,y_t)
    (3) sum(prec(t-24h),...,prec(t)) > sensor_meas_depth * sensor_accuracy * soil_porosity

    NOTE1: np.nan entries in the input precipitation series will be regarded as susipicious and the test will be
    omited for every 24h interval including a np.nan entrie in the original precipitation sampling rate.
    Only entry "0" will be regarded as denoting "No Rainfall".

    NOTE2: The function wont test any values that are flagged suspicious anyway - this may change in a future version.


    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision.
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
    :param std_factor_range:            Offset String. Denotes the range over witch the standart deviation is obtained,
                                        to test condition [2]. (Should be a multiple of the sampling rate)
    :param raise_reference:             Offset String. Denotes the distance to the datapoint, relatively to witch
                                        it is decided if the current datapoint is a raise or not. Equation [1].
                                        It defaults to None. When None is passed, raise_reference is just the sample
                                        rate of the data. Any raise reference must be a multiple of the (intended)
                                        sample rate and below std_factor_range.
    """

    dataseries, moist_rate = retrieveTrustworthyOriginal(data, field, flagger)

    # retrieve input sampling rate (needed to translate ref and data rates into each other):
    input_rate = estimateSamplingRate(data.index)
    refseries, ref_rate = retrieveTrustworthyOriginal(data, prec_reference, flagger)
    # abort processing if any of the measurement series has no valid entries!
    if moist_rate is np.nan:
        return data, flagger
    if ref_rate is np.nan:
        return data, flagger

    # get 24 h prec. monitor (this makes last-24h-rainfall-evaluation independent from preceeding entries)
    prec_count = refseries.rolling(window="1D").apply(
        lambda x: x.sum(skipna=False), raw=False
    )
    # upsample with zeros to input data sampling rate (we want to project the daysums onto the dataseries grid to
    # prepare for use of rolling:):
    prec_count = prec_count.resample(input_rate).pad()

    # now we can: project precipitation onto dataseries sampling (and stack result to be able to apply df.rolling())
    eval_frame = (
        pd.merge(dataseries, prec_count, how="left", left_index=True, right_index=True)
        .stack(dropna=False)
        .reset_index()
    )

    # following reshaping operations make all columns available to a function applied on rolling windows (rolling will
    # only eat one column of a dataframe at a time and doesnt like multi indexes as well)
    ef = eval_frame[0]
    ef.index = eval_frame["level_0"]

    if raise_reference is None:
        raise_reference = 1
    else:
        raise_reference = int(offset2periods(raise_reference, moist_rate))
    # make raise and std. dev tester function (returns False for values that
    # should be flagged bad and True respectively. (must be this way, since np.nan gets casted to True)))
    def _precTest(x, std_fac=std_factor, raise_ref=raise_reference):
        x_moist = x[0::2]
        x_rain = x[1::2]
        if x_moist[-1] > x_moist[(-1 - raise_ref)]:
            if (x_moist[-1] - x_moist[0]) > std_fac * x_moist.std():
                return ~(
                    x_rain[-1] <= (sensor_meas_depth * soil_porosity * sensor_accuracy)
                )
            else:
                return True
        else:
            return True

    # rolling.apply should only get active every second entrie of the stacked frame,
    # so periods per window have to be calculated,
    # (this gives sufficiant conditian since window size controlls daterange:)

    periods = int(2 * offset2periods(std_factor_range, moist_rate))
    invalid_raises = (
        ~ef.rolling(window="1D", closed="both", min_periods=periods)
        .apply(_precTest, raw=False)
        .astype(bool)
    )
    # undo stacking (only every second entrie actually is holding an information:
    invalid_raises = invalid_raises[1::2]
    # retrieve indices referring to values-to-be-flagged-bad
    invalid_indices = invalid_raises.index[invalid_raises]
    # set Flags
    flagger = flagger.setFlags(field, invalid_indices, **kwargs)
    return data, flagger


@register("soilMoisture_constant")
def flagSoilMoistureByConstantsDetection(
    data,
    field,
    flagger,
    plateau_window_min="12h",
    plateau_var_limit=0.0005,
    rainfall_window="12h",
    filter_window_size="3h",
    i_start_infimum=0.0025,
    i_end_supremum=0,
    data_max_tolerance=0.95,
    **kwargs
):

    """Function is not ready to use yet: we are waiting for response from the author of [Paper] in order of getting
    able to exclude some sources of confusion.

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    """
    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)

    # abort processing if original series has no valid entries!
    if data_rate is np.nan:
        return data, flagger

    # get data max
    data_max = dataseries.max()

    min_periods = int(offset2periods(plateau_window_min, data_rate))
    # identify minimal plateaus:
    plateaus = dataseries.rolling(window=plateau_window_min).apply(
        lambda x: (x.var() > plateau_var_limit) | (x.size < min_periods), raw=False
    )
    plateaus = ~plateaus.astype(bool)

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flagger

    plateaus_reverse = pd.Series(np.flip(plateaus.values), index=plateaus.index)
    reverse_check = (
        plateaus_reverse.rolling(window=plateau_window_min)
        .apply(lambda x: True if True in x.values else False, raw=False)
        .astype(bool)
    )
    plateaus = pd.Series(np.flip(reverse_check.values), index=plateaus.index)

    # reverse the reversed ts and transform to dataframe, filter for consecutive timestamp values:
    plateaus = pd.DataFrame(
        {"date": dataseries.index, "mask": np.flip(plateaus.values)},
        index=dataseries.index,
    )
    plateaus = plateaus[plateaus["mask"] == True].drop("mask", axis=1)
    seperator_stair = plateaus["date"].diff() != pd.Timedelta(data_rate)
    plateaus["interval_nr"] = seperator_stair.cumsum()
    plateaus = plateaus["interval_nr"]
    invalids = pd.Series([])
    # loop through the intervals to be checked:
    for interval_2_check in range(1, plateaus.max() + 1):
        # how big is the interval?
        interval_delimeter = (
            plateaus[plateaus == interval_2_check].index[-1]
            - plateaus[plateaus == interval_2_check].index[0]
        )

        # slices of the area for the rainfallsearch
        check_start = (
            plateaus[plateaus == interval_2_check].index[0]
            - interval_delimeter
            - pd.Timedelta(rainfall_window)
        )
        check_end = (
            plateaus[plateaus == interval_2_check].index[-1]
            - interval_delimeter
            + pd.Timedelta(rainfall_window)
        )

        # slices to be smoothed and derivated
        smooth_start = check_start - pd.Timedelta(filter_window_size)
        smooth_end = check_end + pd.Timedelta(filter_window_size)
        data_slice = dataseries[smooth_start:smooth_end]

        # calculate first derivative of testing slice:
        smoothing_periods = int(np.ceil(offset2periods(filter_window_size, data_rate)))
        if smoothing_periods % 2 == 0:
            smoothing_periods += 1

        # check if the data slice to be checked is sufficiently big for smoothing options:
        if data_slice.size < smoothing_periods:
            smoothing_periods = data_slice.size
            if smoothing_periods % 2 == 0:
                smoothing_periods -= 1

        # calculate the derivative
        first_deri_series = pd.Series(
            data=savgol_filter(
                data_slice, window_length=smoothing_periods, polyorder=2, deriv=1
            ),
            index=data_slice.index,
        )

        # get test slice
        first_deri_series = first_deri_series[check_start:check_end]
        if first_deri_series.empty:
            continue

        # check some explicit and implicit conditions:
        rainfall_periods = int(offset2periods(rainfall_window, data_rate) * 2)
        if rainfall_periods % 2 == 0:
            rainfall_periods += 1
        maximums = first_deri_series.rolling(
            window=rainfall_periods, center=True, closed="left"
        ).max()
        minimums = first_deri_series.rolling(
            window=rainfall_periods, center=True, closed="left"
        ).min()

        maximums = maximums[maximums > i_start_infimum]
        minimums = minimums[minimums < i_end_supremum]

        if maximums.empty | minimums.empty:
            continue

        i_start_index = maximums.index[0]
        i_end_index = minimums.index[-1]

        if i_start_index > i_end_index:
            continue

        potential_invalid = data_slice[i_start_index:i_end_index]
        # test if the plateau is a high level plateau:
        if potential_invalid.mean() > data_max * data_max_tolerance:
            invalids = pd.concat([invalids, potential_invalid])

    flagger = flagger.setFlag(field, invalids.index, **kwargs)
    return data, flagger
