#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .register import register
import numpy.polynomial.polynomial as poly

from ..lib.tools import (
    inferFrequency,
    retrieveTrustworthyOriginal,
    getPandasVarNames,
    getPandasData,
    offset2seconds,
    checkQCParameters)


@register("polyresmad")
def polyResMad(data, flags, field, flagger, winsz, count=1, deg=1, dx=1, z=3.5, **kwargs):
    d = getPandasData(data, field).copy()
    d = d[d.notna()]
    x = (d.index - d.index.min()).total_seconds().values
    y = d.values

    # checks
    if len(x) < deg + 1:
        raise ValueError(f'deg {deg} to low')
    if deg < 0:
        raise ValueError("deg must be positive")
    if dx <= 0:
        raise ValueError("step size `dx` must be positive and not zero")
    if dx >= winsz and count > 1:
        ValueError("If stepsize `dx` is bigger that the window every value is just seen once, so use count=1")
    if winsz < count * dx:
        raise ValueError(f"Adjust `dx`, `stepsize` or `winsz`. A single data point is "
                         f"seen `winsz / dx = {winsz // dx}` times, but count is set to {count}")

    counters = np.ones(len(x)) * count

    # sliding window loop
    for i in range(0, len(x) - winsz + 1, dx):
        # Indices of the chunk
        chunk = np.arange(i, i + winsz)
        # Exclude points that have been already discarded
        indices = chunk[counters[chunk] > 0]

        # get residual
        xx, yy = x[indices], y[indices]
        coef = poly.polyfit(xx, yy, deg)
        model = poly.polyval(xx, coef)
        residual = yy - model

        # calc mad
        diff = np.abs(residual - np.median(residual))
        mad = np.median(diff)
        zscore = (mad > 0) & (0.6745 * diff > z * mad)

        # count`em in
        goneMad = np.where(zscore)[0]
        counters[indices[goneMad]] -= 1

    outlier = np.where(counters <= 0)[0]
    idx = d.iloc[outlier.tolist()].index

    flags = flagger.setFlags(flags, field, idx, **kwargs)
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
def flagSpikes_Basic(data, flags, field, flagger, thresh=7, tol=0, length=15):
    """
    The Function detects spikes which have a first step > thresh and then a 'plateau' for <= length time steps and come back
    to original value within a tolerance of toler. (something like a rectangular shape, but the 'plateau' does not have to be flat,
    it just needs to exceed the threshold without crossing the value before the spike).
    Returns list with indices of detected spikes.

    Data do not have to be harmonized to frequency.

    The implementation is basically a copy of code, licensed as follows:
    (original) License:
    -------
    This file is part of the UFZ Python package.

    The UFZ Python package is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    The UFZ Python package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with the UFZ Python package (cf. gpl.txt and lgpl.txt).
    If not, see <http://www.gnu.org/licenses/>.

    Copyright 2016 Benjamin Dechant

    The original code (in its last version) can be found here:
    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py


    :param data:
    :param flags:
    :param field:
    :param flagger:
    :param thresh:
    :param toler:
    :param length:
    :return:
    """

    # retrieve data series
    dataseries = data[field].dropna()
    pre_jumps = dataseries.diff(periods=-1).abs() > thresh
    pre_jumps = pre_jumps[pre_jumps]
    to_roll = pre_jumps.reindex(dataseries.index, method='ffill', tolerance=length, fill_value=False).dropna()

    def spike_tester(chunk, pre_jumps, thresh, tol):
        if not chunk.index[0] in pre_jumps.index:
            return 0
        else:
            # signum change!!!
            chunk_stair = (abs(chunk - chunk[0]) < thresh).cumsum()
            first_return = (chunk_stair == 2)
            if first_return.sum() == 0:
                return 0
            if abs(chunk[first_return[first_return].index[0]] - chunk[0]) < tol:
                return (chunk_stair == 1).sum() - 1
            else:
                return 0

    to_flag = dataseries[to_roll].rolling(length, closed='both').apply(spike_tester, args=(pre_jumps, thresh, tol), raw=False)
    # little mess, because goddam rolling doesnt offer label='right' option.... god damn it.
    to_flag.rolling(1).apply(lambda x: to_flag.index.get_loc(x.index[0] - pd.Timedelta('30min'), method='bfill'), raw=False)

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
