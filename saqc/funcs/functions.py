#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


from ..lib.tools import (
    valueRange,
    slidingWindowIndices,
    inferFrequency,
    estimateSamplingRate,
    retrieveTrustworthyOriginal,
    offset2periods,
    offset2seconds,
    checkQCParameters)

from ..dsl import evalExpression
from ..core.config import Params

# NOTE: will be filled by calls to register
FUNC_MAP = {}


def register(name):

    def outer(func):
        func = partial(func, func_name=name)
        FUNC_MAP[name] = func

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
def flagGeneric(data, flags, field, flagger, nodata=np.nan, **kwargs):
    expression = kwargs[Params.FUNC]
    result = evalExpression(expression, flagger,
                            data, flags, field,
                            nodata=nodata)

    result = result.squeeze()

    if np.isscalar(result):
        raise TypeError(f"expression '{expression}' does not return an array")

    if not np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"expression '{expression}' does not return a boolean array")

    fchunk = flagger.setFlag(flags=flags.loc[result, field], **kwargs)

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


@register("Constants_VarianceBased")
def flagConstants_VarianceBased(data, flags, field, flagger, plateau_window_min='12h',
                                         plateau_var_limit=0.0005, **kwargs):

    """Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:

    (1) n > "plateau_interval_min"
    (2) variance(y(t),...,y(t+n) < plateau_var_limit

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
    :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    :param plateau_window_min:          Offset String. Only intervals of minimum size "plateau_window_min" have the
                                        chance to get flagged as constant intervals
    :param plateau_var_limit:           Float. The upper barrier, the variance of an interval mus not exceed, if the
                                        interval wants to be flagged a plateau.
    """


    para_check = checkQCParameters({'data': {'value': data,
                                             'type': [pd.Series, pd.DataFrame],
                                             'tests': {'harmonized': lambda x: pd.infer_freq(x.index) is not None}},
                                    'flags': {'value': flags,
                                              'type': [pd.Series, pd.DataFrame]},
                                    'field': {'value': field,
                                              'type': [str]}}, 'flagConstants_VarianceBased')


    if isinstance(data, pd.Series):
        dataseries, data_rate = retrieveTrustworthyOriginal(data, flags, flagger)
    else:
        dataseries, data_rate = retrieveTrustworthyOriginal(data[field], flags[field], flagger)
        # abort processing if original series has no valid entries!
    if data_rate is np.nan:
        return data, flags

    min_periods = int(offset2periods(plateau_window_min, data_rate))

    # identify minimal plateaus:
    plateaus = dataseries.rolling(window=plateau_window_min).apply(lambda x: (x.var() > plateau_var_limit) |
                                                                             (x.size < min_periods), raw=False)
    plateaus = (~plateaus.astype(bool))

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flags

    # nice reverse trick to cover total interval size
    plateaus_reverse = pd.Series(np.flip(plateaus.values), index=plateaus.index)
    reverse_check = plateaus_reverse.rolling(window=plateau_window_min).apply(
        lambda x: True if True in x.values else False, raw=False).astype(bool)

    # result:
    plateaus = pd.Series(np.flip(reverse_check.values), index=plateaus.index)

    if isinstance(flags, pd.Series):
        flags.loc[plateaus.index, field] = flagger.setFlag(flags.loc[plateaus.index, field], **kwargs)
    else:
        flags.loc[plateaus.index] = flagger.setFlag(flags.loc[plateaus.index], **kwargs)
    return data, flags


@register("Spikes_SpektrumBased")
def flagSpikes_SpektrumBased(data, flags, field, flagger, diff_method='raw', filter_window_size='3h',
                             raise_factor=0.15, dev_cont_factor=0.2, noise_barrier=1, noise_window_size='12h',
                             noise_statistic='CoVar', smooth_poly_order=2, **kwargs):

    """This Function is an generalization of the Spectrum based Spike flagging mechanism as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    Function detects and flags spikes in input data series by evaluating its derivatives and applying some
    conditions to it. A datapoint is considered a spike, if:

    (1) the quotient to its preceeding datapoint exceeds a certain bound
    (controlled by param "raise_factor")
    (2) the quotient of the datas second derivate at the preceeding and subsequent timestamps is close enough to 1.
    (controlled by param "dev_cont_factor")
    (3) the surrounding data is not too noisy. (Coefficient of Variation[+/- noise_window] < 1)
    (controlled by param "noise_barrier")

    Some things you should be conscious about when applying this test:

       NOTE1: You should run less complex tests, especially range-tests, or absolute spike tests previously to this one,
       since the spike check for any potential, unflagged spike, is relatively costly
       (1 x smoothing + 2 x deviating + 2 x condition application).

       NOTE2: Due to inconsistency in the paper that provided the concept of this test [paper:], its not really clear
       weather to use the coefficient of variance or the relative variance for noise testing.
       Since the relative variance was explicitly denoted in the formulas, the function defaults to relative variance,
       but can be switched to coefficient of variance, by assignment to parameter "noise statistic".

       NOTE3: All derivatives in the reference publication are obtained by applying a Savitzky-Golay filter to the data
       before differentiating. For the break detection algorithm in this publication,
       some of the conditions didnt work well with smoothed derivatives.
       This is because smoothing distributes the harshness of breaks and jumps over the
       smoothing window and makes it "smoother".
       Since just taking the differences as derivatives did work well for my empirical data set,
       the parameter "diff_method" defaults to "raw". That means, that derivatives will be obtained by just using the
       differences series.
       You are free of course, to change this parameter to "savgol" and play around with the associated filter options.
       (see parameter description below)



       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision.
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
       :param diff_method:                 String. Method for obtaining dataseries' derivatives.
                                           'raw': Just take series step differences (default)
                                           'savgol': Smooth data with a Savitzky Golay Filter before differentiating.
       :param filter_window_size:          Offset string. Size of the filter window, used to calculate the derivatives.
                                           (relevant only, if: diff_method='savgol')
       :param smooth_poly_order:           Integer. Polynomial order, used for smoothing with savitzk golay filter.
                                           (relevant only, if: diff_method='savgol')
       :param raise_factor:                A float, determinating the bound, the quotient of two consecutive values
                                           has to exceed, to be regarded as potentially spike. A value of 0.x will
                                           trigger the spike test for value y_t, if:
                                           (y_t)/(y_t-1) > 1 + x or:
                                           (y_t)/(y_t-1) < 1 - x.
       :param dev_cont_factor:             A float, determining the interval, the quotient of the datas second derivate
                                           around a potential spike has to be part of, to trigger spike flagging for
                                           this value. A datapoint y_t will pass this spike condition if,
                                           for dev_cont_factor = 0.x, and the second derivative y'' of y, the condition:
                                           1 - x < abs((y''_t-1)/(y''_t+1)) < 1 + x
                                           holds
       :param noise_barrier:               A float, determining the bound, the data noisy-ness around a potential spike
                                           must not exceed, in order to guarantee a justifyed judgement:
                                           Therefor the coefficient selected by parameter noise_statistic (COVA),
                                           of all values within t +/- param "noise_window",
                                           but excluding the point y_t itself, is evaluated and tested
                                           for: COVA < noise_barrier.
       :param noise_window_size:           Offset string, determining the size of the window, the coefficient of variation
                                           is calculated of, to determine data noisy-ness around a potential spike.
                                           The potential spike y_t will be centered in a window of expansion:
                                           [y_t - noise_window_size, y_t + noise_window_size].
       :param noise_statistic:             STRING. Determines, wheather to use
                                           "relative variance" or "coefficient of variation" to check against the noise
                                           barrier.
                                           'CoVar' -> "Coefficient of variation"
                                           'rVar'  -> "relative Variance"
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

    # retrieve noise statistic
    if noise_statistic == 'CoVar':
        noise_func = pd.Series.var
    if noise_statistic == 'rVar':
        noise_func = pd.Series.std

    quotient_series = dataseries/dataseries.shift(+1)
    spikes = (quotient_series > (1 + raise_factor)) | (quotient_series < (1 - raise_factor))
    spikes = spikes[spikes == True]

    # loop through spikes: (loop may sound ugly - but since the number of spikes is supposed to not exceed the
    # thousands for year data - a loop going through all the spikes instances is much faster than
    # a rolling window, rolling all through a stacked year dataframe )

    # calculate some values, repeatedly needed in the course of the loop:

    filter_window_seconds = offset2seconds(filter_window_size)
    smoothing_periods = int(np.ceil((filter_window_seconds / data_rate.n)))
    lower_dev_bound = 1 - dev_cont_factor
    upper_dev_bound = 1 + dev_cont_factor

    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for spike in spikes.index:
        start_slice = spike - pd.Timedelta(filter_window_size)
        end_slice = spike + pd.Timedelta(filter_window_size)

        if diff_method == 'savgol':
            scnd_derivate = savgol_filter(dataseries[start_slice:end_slice],
                                          window_length=smoothing_periods,
                                          polyorder=smooth_poly_order,
                                          deriv=2)

        if diff_method == 'raw':
            scnd_derivative = dataseries[start_slice:end_slice].diff().diff()

        length = scnd_derivate.size
        test_ratio_1 = np.abs(scnd_derivate[int((length-1) / 2)] / scnd_derivate[int((length+1) / 2)])

        if lower_dev_bound < test_ratio_1 < upper_dev_bound:
            # apply noise condition:
            start_slice = spike - pd.Timedelta(noise_window_size)
            end_slice = spike + pd.Timedelta(noise_window_size)
            test_slice = dataseries[start_slice:end_slice].drop(spike)
            test_ratio_2 = np.abs(noise_func(test_slice) / test_slice.mean())
            # not a spike, we want to flag, if condition not satisfied:
            if test_ratio_2 > noise_barrier:
                spikes[spike] = False

        # not a spike, we want to flag, if condition not satisfied
        else:
            spikes[spike] = False

    spikes = spikes[spikes == True]

    if isinstance(flags, pd.Series):
        flags.loc[spikes.index, field] = flagger.setFlag(flags.loc[spikes.index, field], **kwargs)
    else:
        flags.loc[spikes.index] = flagger.setFlag(flags.loc[spikes.index], **kwargs)
    return data, flags



@register("Breaks_SpektrumBased")
def flagBreaks_SpektrumBased(data, flags, field, flagger, diff_method='raw', filter_window_size='3h',
                             rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                             first_der_window_size='12h', scnd_der_ratio_margin_1=0.05,
                             scnd_der_ratio_margin_2=10, smooth_poly_order=2, **kwargs):

    """ This Function is an generalization of the Spectrum based break flagging mechanism as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    The function flags breaks (jumps/drops) in input measurement series by evaluating its derivatives.
    A measurement y_t is flagged a, break, if:

    (1) y_t is changing relatively to its preceeding value by at least (100*rel_change_rate_min) percent
    (2) y_(t-1) is difffering from its preceeding value, by a margin of at least "abs_change_min"
    (3) Absolute first derivative |(y_t)'| has to be at least "first_der_factor" times as big as the arithmetic middle
        over all the first derivative values within a 2 times "first_der_window_size" hours window, centered at t.
    (4) The ratio of the second derivatives at t and t+1 has to be "aproximately" 1.
        ([1-scnd__der_ration_margin_1, 1+scnd_ratio_margin_1])
    (5) The ratio of the second derivatives at t+1 and t+2 has to be larger than scnd_der_ratio_margin_2

    NOTE 1: As no reliable statement about the plausibility of the meassurements before and after the jump is possible,
    only the jump itself is flagged. For flagging constant values following upon a jump, use a flagConstants test.

    NOTE 2: All derivatives in the reference publication are obtained by applying a Savitzky-Golay filter to the data
    before differentiating. However, i was not able to reproduce satisfaction of all the conditions for synthetically
    constructed breaks.
    Especially condition [4] and [5]! This is because smoothing distributes the harshness of the break over the
    smoothing window. Since just taking the differences as derivatives did work well for my empirical data set,
    the parameter "diff_method" defaults to "raw". That means, that derivatives will be obtained by just using the
    differences series.
    You are free of course, to change this parameter to "savgol" and play around with the associated filter options.
    (see parameter description below)




       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision (skips allowed).
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
       :param diff_method:                 String. Method for obtaining dataseries' derivatives.
                                           'raw': Just take series step differences (default)
                                           'savgol': Smooth data with a Savitzky Golay Filter before differentiating.
       :param filter_window_size:          Offset string. Size of the filter window, used to calculate the derivatives.
                                           (relevant only, if: diff_method='savgol')
       :param smooth_poly_order:           Integer. Polynomial order, used for smoothing with savitzk golay filter.
                                           (relevant only, if: diff_method='savgol')
       :param rel_change_rate_min          Float in [0,1]. See (1) of function descritpion above to learn more
       :param abs_change_min               Float > 0. See (2) of function descritpion above to learn more.
       :param first_der_factor             Float > 0. See (3) of function descritpion above to learn more.
       :param first_der_window_size        Offset_String. See (3) of function description to learn more.
       :param scnd_der_ratio_margin_1      Float in [0,1]. See (4) of function descritpion above to learn more.
       :param scnd_der_ratio_margin_2      Float in [0,1]. See (5) of function descritpion above to learn more.
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
    breaks = (abs_change > abs_change_min) & (abs_change / dataseries > rel_change_rate_min)
    breaks = breaks[breaks == True]

    # First derivative criterion
    smoothing_periods = int(np.ceil(offset2periods(filter_window_size, data_rate)))
    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for brake in breaks.index:
        # slice out slice-to-be-filtered (with some safety extension of 3 hours)
        slice_start = brake - pd.Timedelta(first_der_window_size) -pd.Timedelta('3h')
        slice_end = brake + pd.Timedelta(first_der_window_size) + pd.Timedelta('3h')
        data_slice = dataseries[slice_start:slice_end]

        # obtain first derivative:
        if diff_method == 'savgol':
            first_deri_series = pd.Series(data=savgol_filter(data_slice,
                                          window_length=smoothing_periods,
                                          polyorder=smooth_poly_order,
                                          deriv=1),
                                          index=data_slice.index)
        if diff_method == 'raw':
            first_deri_series = data_slice.diff()

        # condition constructing and testing:
        test_slice = first_deri_series[brake - pd.Timedelta(first_der_window_size):
                                       brake + pd.Timedelta(first_der_window_size)]

        test_sum = abs((test_slice.sum()*first_der_factor) / test_slice.size)

        if abs(first_deri_series[brake]) > test_sum:
            # second derivative criterion:
            slice_start = brake - pd.Timedelta('3h')
            slice_end = brake + pd.Timedelta('3h')
            data_slice = data_slice[slice_start:slice_end]

            # obtain second derivative:
            if diff_method == 'savgol':
                second_deri_series = pd.Series(data=savgol_filter(data_slice,
                                               window_length=smoothing_periods,
                                               polyorder=smooth_poly_order,
                                               deriv=2),
                                               index=data_slice.index)
            if diff_method == 'raw':
                second_deri_series = data_slice.diff().diff()

            # criterion evaluation:
            first_second = (1 - scnd_der_ratio_margin_1) \
                           < abs((second_deri_series.shift(-1)[brake] / second_deri_series.shift(-2)[brake])) \
                           < 1 + scnd_der_ratio_margin_1

            second_second = abs(second_deri_series.shift(-1)[brake] / second_deri_series.shift(-2)[brake]) \
                            > scnd_der_ratio_margin_2

            if (~ first_second) | (~ second_second):
                breaks[brake] = False

        else:
            breaks[brake] = False

    breaks = breaks[breaks == True]

    if isinstance(flags, pd.Series):
        flags.loc[breaks.index] = flagger.setFlag(flags.loc[breaks.index], **kwargs)
    else:
        flags.loc[breaks.index, field] = flagger.setFlag(flags.loc[breaks.index, field], **kwargs)

    return data, flags


@register("SoilMoistureSpikes")
def flagSoilMoistureSpikes(data, flags, field, flagger, filter_window_size='3h',
                             raise_factor=0.15, dev_cont_factor=0.2, noise_barrier=1, noise_window_size='12h',
                             noise_statistic='CoVar', **kwargs):

    """
    The Function provides just a call to flagSpikes_SpektrumBased, with parameter defaults, that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.
    """

    return flagSpikes_SpektrumBased(data, flags, field, flagger, filter_window_size=filter_window_size,
                                    raise_factor=raise_factor, dev_cont_factor=dev_cont_factor,
                                    noise_barrier=noise_barrier, noise_window_size=noise_window_size,
                                    noise_statistic=noise_statistic, **kwargs)


@register("SoilMoistureBreaks")
def flagSoilMoistureBreaks(data, flags, field, flagger, diff_method='raw', filter_window_size='3h',
                           rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                           first_der_window_size='12h', scnd_der_ratio_margin_1=0.05,
                           scnd_der_ratio_margin_2=10, smooth_poly_order=2, **kwargs):

    """
    The Function provides just a call to flagBreaks_SpektrumBased, with parameter defaults that refer to:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    """
    return flagBreaks_SpektrumBased(data, flags, field, flagger, diff_method=diff_method,
                                    filter_window_size=filter_window_size,
                                    rel_change_rate_min=rel_change_rate_min, abs_change_min=abs_change_min,
                                    first_der_factor=first_der_factor, first_der_window_size=first_der_window_size,
                                    scnd_der_ratio_margin_1=scnd_der_ratio_margin_1,
                                    scnd_der_ratio_margin_2=scnd_der_ratio_margin_2,
                                    smooth_poly_order=smooth_poly_order, **kwargs)


@register("SoilMoistureByFrost")
def flagSoilMoistureBySoilFrost(data, flags, field, flagger, soil_temp_reference, tolerated_deviation='1h',
                                frost_level=0, **kwargs):

    """This Function is an implementation of the soil temperature based Soil Moisture flagging, as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    All parameters default to the values, suggested in this publication.

    Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference
                                        series. Data must be indexed by a datetime series.
    :param flags:                       A dataframe holding the flags/flag-entries of "data"
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

    # SaQC policy: Only data that has been flagged by at least one test is allowed to be referred to:
    if soil_temp_reference not in flags.columns:
        logging.warning('The reference variable {} is either not part of the passed data frame, or the value is not '
                        'registered to the flags frame. To register it to the flags frame and thus, make it available '
                        'for reference within tests, you need to run at least one single targeted test on it. '
                        'The test will be skipped.'.format(soil_temp_reference))
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


@register("SoilMoistureByPrecipitation")
def flagSoilMoistureByPrecipitationEvents(data, flags, field, flagger, prec_reference, sensor_meas_depth=0,
                                          sensor_accuracy=0, soil_porosity=0, std_factor=2, std_factor_range='24h',
                                          raise_reference=None, **kwargs):

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
    :param std_factor_range:            Offset String. Denotes the range over witch the standart deviation is obtained,
                                        to test condition [2]. (Should be a multiple of the sampling rate)
    :param raise_reference:             Offset String. Denotes the distance to the datapoint, relatively to witch
                                        it is decided if the current datapoint is a raise or not. Equation [1].
                                        It defaults to None. When None is passed, raise_reference is just the sample
                                        rate of the data. Any raise reference must be a multiple of the (intended)
                                        sample rate and below std_factor_range.
    """

    # SaQC policy: Only data that has been flagged by at least one test is allowed to be referred to:

    if prec_reference not in flags.columns:
        logging.warning(
            'The reference variable {} is either not part of the passed data frame, or the value is not '
            'registered to the flags frame. To register it to the flags frame and thus, make it available '
            'for reference within tests, you need to run at least one single targeted test on it. '
            'The test will be skipped.'.format(prec_reference))
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

    if raise_reference is None:
        raise_reference = 1
    else:
        raise_reference = int(offset2periods(raise_reference, moist_rate))
    # make raise and std. dev tester function (returns False for values that
    # should be flagged bad and True respectively. (must be this way, since np.nan gets casted to True)))
    def prec_test(x, std_fac=std_factor, raise_ref=raise_reference):
        x_moist = x[0::2]
        x_rain = x[1::2]
        if x_moist[-1] > x_moist[(-1-raise_ref)]:
            if (x_moist[-1] - x_moist[0]) > std_fac*x_moist.std():
                return ~(x_rain[-1] <= (sensor_meas_depth*soil_porosity*sensor_accuracy))
            else:
                return True
        else:
            return True

    # rolling.apply should only get active every second entrie of the stacked frame,
    # so periods per window have to be calculated,
    # (this gives sufficiant conditian since window size controlls daterange:)

    periods = int(2*offset2periods(std_factor_range, moist_rate))
    invalid_raises = ~ef.rolling(window='1D', closed='both', min_periods=periods)\
        .apply(prec_test, raw=False).astype(bool)
    # undo stacking (only every second entrie actually is holding an information:
    invalid_raises = invalid_raises[1::2]
    # retrieve indices referring to values-to-be-flagged-bad
    invalid_indices = invalid_raises.index[invalid_raises]
    # set Flags
    flags.loc[invalid_indices, field] = flagger.setFlag(flags.loc[invalid_indices, field], **kwargs)
    return data, flags


def flagSoilMoistureByConstantsDetection(data, flags, field, flagger, plateau_window_min='12h',
                                         plateau_var_limit=0.0005, rainfall_window='12h', filter_window_size='3h',
                                         i_start_infimum=0.0025, i_end_supremum=0, data_max_tolerance=0.95, **kwargs):

    """Function is not ready to use yet: we are waiting for response from the author of [Paper] in order of getting
    able to exclude some sources of confusion.

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
    :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    """
    if isinstance(data, pd.Series):
        dataseries, data_rate = retrieveTrustworthyOriginal(data, flags, flagger)
    else:
        dataseries, data_rate = retrieveTrustworthyOriginal(data[field], flags[field], flagger)
        # abort processing if original series has no valid entries!
    if data_rate is np.nan:
        return data, flags

    # get data max
    data_max = dataseries.max()

    min_periods = int(offset2periods(plateau_window_min, data_rate))
    # identify minimal plateaus:
    plateaus = dataseries.rolling(window=plateau_window_min).apply(lambda x: (x.var() > plateau_var_limit) | (x.size < min_periods), raw=False)
    plateaus = (~plateaus.astype(bool))

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flags

    plateaus_reverse = pd.Series(np.flip(plateaus.values), index=plateaus.index)
    reverse_check = plateaus_reverse.rolling(window=plateau_window_min).apply(lambda x:
                                                                              True if True in x.values else False,
                                                                              raw=False).astype(bool)
    plateaus = pd.Series(np.flip(reverse_check.values), index=plateaus.index)


    # reverse the reversed ts and transform to dataframe, filter for consecutive timestamp values:
    plateaus = pd.DataFrame({'date': dataseries.index, 'mask': np.flip(plateaus.values)}, index=dataseries.index)
    plateaus = plateaus[plateaus['mask'] == True].drop('mask',axis=1)
    seperator_stair = plateaus['date'].diff() != pd.Timedelta(data_rate)
    plateaus['interval_nr'] = seperator_stair.cumsum()
    plateaus = plateaus['interval_nr']
    invalids = pd.Series([])
    # loop through the intervals to be checked:
    for interval_2_check in range(1, plateaus.max()+1):
        # how big is the interval?
        interval_delimeter = plateaus[plateaus==interval_2_check].index[-1] - \
                             plateaus[plateaus==interval_2_check].index[0]

        # slices of the area for the rainfallsearch
        check_start = plateaus[plateaus==interval_2_check].index[0] - interval_delimeter - pd.Timedelta(rainfall_window)
        check_end = plateaus[plateaus==interval_2_check].index[-1] - interval_delimeter + pd.Timedelta(rainfall_window)

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
        first_deri_series = pd.Series(data=savgol_filter(data_slice,
                                      window_length=smoothing_periods,
                                      polyorder=2,
                                      deriv=1),
                                      index=data_slice.index)

        # get test slice
        first_deri_series = first_deri_series[check_start:check_end]
        if first_deri_series.empty:
            continue

        # check some explicit and implicit conditions:
        rainfall_periods = int(offset2periods(rainfall_window, data_rate)*2)
        if rainfall_periods % 2 == 0:
            rainfall_periods += 1
        maximums = first_deri_series.rolling(window=rainfall_periods, center=True, closed='left').max()
        minimums = first_deri_series.rolling(window=rainfall_periods, center=True, closed='left').min()

        maximums=maximums[maximums > i_start_infimum]
        minimums=minimums[minimums < i_end_supremum]

        if maximums.empty | minimums.empty:
            continue

        i_start_index = maximums.index[0]
        i_end_index = minimums.index[-1]

        if i_start_index > i_end_index:
            continue

        potential_invalid = data_slice[i_start_index:i_end_index]
        # test if the plateau is a high level plateau:
        if potential_invalid.mean() > data_max*data_max_tolerance:
            invalids = pd.concat([invalids, potential_invalid])

    if isinstance(flags, pd.Series):
        flags.loc[invalids.index] = flagger.setFlag(flags.loc[invalids.index], **kwargs)
    else:
        flags.loc[invalids.index, field] = flagger.setFlag(flags.loc[invalids.index, field], **kwargs)

    return data, flags
