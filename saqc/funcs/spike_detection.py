#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from .register import register
import numpy.polynomial.polynomial as poly

from ..lib.tools import (
    inferFrequency,
    retrieveTrustworthyOriginal,
    getPandasVarNames,
    getPandasData,
    offset2seconds,
    checkQCParameters,
    slidingWindowIndices)


@register("sliding_outlier")
def slidingOutlier(data, flags, field, flagger, winsz, dx, count=1, deg=1, z=3.5, method='modZ', **kwargs):
    """ A outlier detection in a sliding window. The method for detection can be a simple Z-score or the more robust
    modified Z-score, as introduced here [1].

    The steps are:
    1.  a window of size `winsz` is cut from the data
    2.  the data is fit by a polynomial of the given degree `deg`
    3.  the outlier `method` detect potential outlier
    4.  the window is continued by `dx` to the next data-slot.
    5.  processing continue at 1. until end of data.
    6.  all potential outlier, that are detected `count`-many times, are promoted to real outlier and flagged by the `flagger`

    :param data:        pandas dataframe. holding the data
    :param flags:       pandas dataframe. holding the flags
    :param field:       fieldname in `data` and `flags`, which holds the relevant infos
    :param flagger:     flagger.
    :param winsz:       int or time-offset string (see [2]). The size of the window the outlier detection is run in. default: 1h
    :param dx:          int or time-offset string (see [2]). Stepsize the window is set further. default: 1h
    :param method:      str. `modZ`  or `zscore`. see [1] at section `Z-Scores and Modified Z-Scores`
    :param count:       int. this many times, a datapoint needs to be detected in different windows, to be finally
                        flagged as outlier
    :param deg:         int. The degree for the polynomial fit, to calculate the residuum
    :param z:           float. the value the (mod.) Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])

    Links:
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    [2] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    """

    use_offset = False
    dx_s = dx
    winsz_s = winsz
    # check param consistency
    if isinstance(winsz, str) or isinstance(dx, str):
        if isinstance(winsz, str) and isinstance(dx, str):
            use_offset = True
            dx_s = offset2seconds(dx)
            winsz_s = offset2seconds(winsz)
        else:
            raise TypeError(f"`winsz` and `dx` must both be an offset or both be numeric, {winsz} and {dx} was passed")

    # check params
    if deg < 0:
        raise ValueError("deg must be positive")
    if z < 0:
        raise ValueError("z must be positive")
    if count <= 0:
        raise ValueError("count must be positive and not zero")

    if dx_s >= winsz_s and count == 1:
        pass
    elif dx_s >= winsz_s and count > 1:
        ValueError("If stepsize `dx` is bigger that the window-size, every value is seen just once, so use count=1")
    elif count > winsz_s // dx_s:
        raise ValueError(f"Adjust `dx`, `stepsize` or `winsz`. A single data point is "
                         f"seen `floor(winsz / dx) = {winsz_s // dx_s}` times, but count is set to {count}")

    # prepare the method
    if method == 'modZ':
        def calc_(residual):
            diff = np.abs(residual - np.median(residual))
            mad = np.median(diff)
            return (mad > 0) & (0.6745 * diff > z * mad)
    elif method == 'zscore':
        def calc_(residual):
            score = zscore(residual, ddof=1)
            return np.abs(score) > z
    else:
        raise NotImplementedError
    method = calc_

    # prepare data, work on numpy arrays for the fulfilling pleasure of performance
    d = data[field].dropna()
    all_indices = np.arange(len(d.index))
    x = (d.index - d.index[0]).total_seconds().values
    y = d.values
    counters = np.full(len(d.index), count)

    if use_offset:
        loopfun = slidingWindowIndices
    else:
        def loopfun (arr, wsz, step):
            for i in range(0, len(arr) - wsz + 1, step):
                yield i, i + wsz


    for start, end in loopfun(d.index, winsz, dx):
        # mask points that have been already discarded
        mask = counters[start:end] > 0
        indices = all_indices[all_indices[start:end][mask]]
        xchunk = x[indices]
        ychunk = y[indices]

        if xchunk.size == 0:
            continue

        # get residual
        coef = poly.polyfit(xchunk, ychunk, deg)
        model = poly.polyval(xchunk, coef)
        residual = ychunk - model

        score = method(residual)

        # count`em in
        goneMad = score.nonzero()[0]
        counters[indices[goneMad]] -= 1

    outlier = np.where(counters <= 0)[0]
    loc = d[outlier].index
    flags = flagger.setFlags(flags, field, loc=loc, **kwargs)
    return data, flags


@register("mad")
def flagMad(data, flags, field, flagger, length, z=3.5, freq=None, **kwargs):
    """ The function represents an implementation of the modyfied Z-score outlier detection method, as introduced here:

    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    The test needs the input data to be harmonized to an equidustant time stamp series (=have frequencie))

    :param data:        The pandas dataframe holding the data-to-be flagged.
                        Data must be indexed by a datetime series and be harmonized onto a
                        time raster with seconds precision.
    :param flags:       A dataframe holding the flags/flag-entries associated with "data".
    :param field:       Fieldname of the Soil moisture measurements field in data.
    :param flagger:     A flagger - object. (saqc.flagger.X)
    :param length:      Offset String. Denoting the windows size that that th "Z-scored" values have to lie in.
    :param z:           Float. The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
    :param freq:        Frequencie.
    """

    d = getPandasData(data, field).copy()
    freq = inferFrequency(d) if freq is None else freq
    if freq is None:
        raise ValueError("freqency cannot inferred, provide `freq` as a param to mad().")
    winsz = int(pd.to_timedelta(length) / freq)
    median = d.rolling(window=winsz, center=True, closed='both').median()
    diff = abs(d - median)
    mad = diff.rolling(window=winsz, center=True, closed='both').median()
    mask = (mad > 0) & (0.6745 * diff > z * mad)

    flags = flagger.setFlags(flags, field, mask, **kwargs)
    return data, flags

@register("Spikes_Basic")
def flagSpikes_Basic(data, flags, field, flagger, thresh=7, tol=0, length='15min', **kwargs):
    """
    A basic outlier test that is designed to work for harmonized and not harmonized data.

    Values x(n), x(n+1), .... , x(n+k) of a timeseries x are considered spikes, if

    (1) |x(n-1) - x(n + s)| > "thresh", for all s in [0,1,2,...,k]

    (2) |x(n-1) - x(n+k+1)| < tol

    (3) |x(n-1).index - x(n+k+1).index| < length

    Note, that this definition of a "spike" not only includes one-value outliers, but also plateau-ish value courses.

    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:

    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py


    :param data:    Pandas-like. The pandas dataframe holding the data-to-be flagged.
    :param flags:   pd.Dataframe. A dataframe holding the flags/flag-entries associated with "data".
    :param field:   String. Fieldname of the data column to be tested.
    :param flagger: saqc.flagger. A flagger - object.
    :param thresh:  Float. The lower bound for a value jump, to be considered as initialising a spike.
                    (see condition (1) in function description).
    :param tol:   Float. Tolerance value.  (see condition (2) in function description)
    :param length:  Offset String. The time span in wich the values of a spikey course have to return to the normal
                    value course (see condition (3) in function description).
    :return:
    """

    # retrieve data series
    dataseries = getPandasData(data, field).copy().dropna()
    # get all the entries preceding a significant jump
    pre_jumps = dataseries.diff(periods=-1).abs() > thresh
    pre_jumps = pre_jumps[pre_jumps]
    # get all the entries preceeding a significant jump and its successors within "length" range
    to_roll = pre_jumps.reindex(dataseries.index, method='ffill', tolerance=length, fill_value=False).dropna()

    # define spike testing function to roll with:
    def spike_tester(chunk, pre_jumps_index, thresh, tol):
        if not chunk.index[-1] in pre_jumps_index:
            return 0
        else:
            # signum change!!!
            chunk_stair = (abs(chunk - chunk[-1]) < thresh)[::-1].cumsum()
            first_return = chunk_stair[(chunk_stair == 2)]
            if first_return.sum() == 0:
                return 0
            if abs(chunk[first_return.index[0]] - chunk[-1]) < tol:
                return (chunk_stair == 1).sum() - 1
            else:
                return 0

    # since .rolling does neither support windows, defined by left starting points, nor rolling over monotonically
    # decreasing indices, we have to trick the method by inverting the timeseries and transforming the resulting index
    # to pseudo-increase.
    to_roll = dataseries[to_roll]
    original_index = to_roll.index
    to_roll = to_roll[::-1]
    pre_jump_reversed_index = to_roll.index[0] - pre_jumps.index
    to_roll.index = to_roll.index[0] - to_roll.index

    # now lets roll:
    to_roll = to_roll.rolling(length, closed='both').\
        apply(spike_tester, args=(pre_jump_reversed_index, thresh, tol), raw=False).astype(int)
    # reconstruct original index and sequence
    to_roll = to_roll[::-1]
    to_roll.index = original_index
    to_write = to_roll[to_roll != 0]
    to_flag = pd.Index([])
    # here comes a loop...):
    for row in to_write.iteritems():
        loc = to_roll.index.get_loc(row[0])
        to_flag = to_flag.append(to_roll.iloc[loc+1:loc+row[1]+1].index)

    to_flag = to_flag.drop_duplicates(keep='first')
    flags = flagger.setFlags(flags, field, to_flag, **kwargs)

    return data, flags

@register("Spikes_SpektrumBased")
def flagSpikes_SpektrumBased(data, flags, field, flagger, filter_window_size='3h',
                             raise_factor=0.15, dev_cont_factor=0.2, noise_barrier=1, noise_window_size='12h',
                             noise_statistic='CoVar', smooth_poly_order=2, **kwargs):
    """
    This Function is a generalization of the Spectrum based Spike flagging mechanism as presented in:

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




       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision.
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
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
       :param noise_window_size:           Offset string, determining the size of the window, the coefficient of
                                           variation is calculated of, to determine data noisy-ness around a potential
                                           spike.
                                           The potential spike y_t will be centered in a window of expansion:
                                           [y_t - noise_window_size, y_t + noise_window_size].
       :param noise_statistic:             STRING. Determines, wheather to use
                                           "relative variance" or "coefficient of variation" to check against the noise
                                           barrier.
                                           'CoVar' -> "Coefficient of variation"
                                           'rVar'  -> "relative Variance"
    """

    para_check_1 = checkQCParameters({'data': {'value': data,
                                               'type': [pd.Series, pd.DataFrame],
                                               'tests': {'harmonized': lambda x: pd.infer_freq(x.index) is not None}},
                                      'flags': {'value': flags,
                                                'type': [pd.Series, pd.DataFrame]},
                                      'field': {'value': field,
                                                'type': [str],
                                                'tests': {'scheduled in data': lambda x: x in
                                                                                         getPandasVarNames(data)}}},
                                     kwargs['func_name'])

    dataseries, data_rate = retrieveTrustworthyOriginal(getPandasData(data, field), getPandasData(flags, field),
                                                        flagger)

    para_check_2 = checkQCParameters({'noise_statistic': {'value': noise_statistic,
                                                          'member': ['CoVar', 'rVar']},
                                      'filter_window_size': {'value': filter_window_size,
                                                             'type': [str],
                                                             'tests': {'Valid Offset String': lambda x: pd.Timedelta(
                                                                 x).total_seconds() % 1 == 0}},
                                      'noise_window_size': {'value': noise_window_size,
                                                            'type': [str],
                                                            'tests': {'Valid Offset String': lambda x: pd.Timedelta(
                                                                x).total_seconds() % 1 == 0}},
                                      'smooth_poly_order': {'value': smooth_poly_order,
                                                            'type': [int],
                                                            'range': [0, np.inf]},
                                      'raise_factor': {'value': raise_factor,
                                                       'type': [int, float],
                                                       'range': [0, 1]},
                                      'noise_barrier': {'value': noise_barrier,
                                                        'type': [int, float],
                                                        'range': [0, np.inf]},
                                      'dev_cont_factor': {'value': dev_cont_factor,
                                                          'type': [int, float],
                                                          'range': [0, 1]}},
                                     kwargs['func_name'])

    # retrieve data series input at its original sampling rate
    # (Note: case distinction for pure series input to avoid error resulting from trying to access pd.Series[field]
    if (para_check_1 < 0) | (para_check_2 < 0):
        logging.warning('test {} will be skipped because not all input parameters satisfied '
                        'the requirements'.format(kwargs['func_name']))
        return data, flags

    # retrieve noise statistic
    if noise_statistic == 'CoVar':
        noise_func = pd.Series.var
    if noise_statistic == 'rVar':
        noise_func = pd.Series.std

    quotient_series = dataseries / dataseries.shift(+1)
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

        scnd_derivate = savgol_filter(dataseries[start_slice:end_slice],
                                      window_length=smoothing_periods,
                                      polyorder=smooth_poly_order,
                                      deriv=2)

        length = scnd_derivate.size
        test_ratio_1 = np.abs(scnd_derivate[int((length - 1) / 2)] / scnd_derivate[int((length + 1) / 2)])

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

    flags = flagger.setFlags(flags, field, spikes.index, **kwargs)
    return data, flags
