#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.optimize import curve_fit
from saqc.funcs.register import register
import numpy.polynomial.polynomial as poly
import numba
import saqc.lib.ts_operators as ts_ops
from saqc.lib.tools import retrieveTrustworthyOriginal, offset2seconds, slidingWindowIndices, findIndex, composeFunction


@register()
def spikes_flagOddWater(
    data,
    field,
    flagger,
    fields,
    trafo="normScale",
    alpha=0.05,
    bin_frac=10,
    n_neighbors=2,
    iter_start=0.5,
    scoring_method="kNNMaxGap",
    lambda_estimator="gap_average",
    **kwargs,
):

    trafo = composeFunction(trafo.split(","))
    # data fransformation/extraction
    val_frame = trafo(data[fields[0]])

    for var in fields[1:]:
        val_frame = pd.merge(val_frame, trafo(data[var]), how="outer", left_index=True, right_index=True)

    data_len = val_frame.index.size
    val_frame.dropna(inplace=True)

    # KNN calculation
    kNNfunc = getattr(ts_ops, scoring_method)
    resids = kNNfunc(val_frame.values, n_neighbors=n_neighbors, algorithm="ball_tree")

    # sorting
    sorted_i = resids.argsort()
    resids = resids[sorted_i]

    # iter_start

    if lambda_estimator == "gap_average":
        sample_size = resids.shape[0]
        gaps = np.append(0, np.diff(resids))
        tail_size = int(max(min(50, np.floor(sample_size / 4)), 2))
        tail_indices = np.arange(2, tail_size + 1)
        i_start = int(max(np.floor(sample_size * iter_start), 1) + 1)
        sum(tail_indices / (tail_size - 1) * gaps[i_start - tail_indices + 1])
        ghat = np.array([np.nan] * sample_size)
        for i in range(i_start - 1, sample_size):
            ghat[i] = sum(tail_indices / (tail_size - 1) * gaps[i - tail_indices + 1])

        log_alpha = np.log(1 / alpha)
        for iter_index in range(i_start - 1, sample_size):
            if gaps[iter_index] > log_alpha * ghat[iter_index]:
                break
    else:
        # (estimator == 'exponential_fit')
        iter_index = int(np.floor(resids.size * iter_start))
        # initialize condition variables:
        crit_val = np.inf
        test_val = 0
        neg_log_alpha = -np.log(alpha)

        # define exponential dist density function:
        def fit_function(x, lambd):
            return lambd * np.exp(-lambd * x)

        # initialise sampling bins
        binz = np.linspace(resids[0], resids[-1], 10 * int(np.ceil(data_len / bin_frac)))
        binzenters = np.array([0.5 * (binz[i] + binz[i + 1]) for i in range(len(binz) - 1)])
        # inititialize full histogram:
        full_hist, binz = np.histogram(resids, bins=binz)
        # check if start index is sufficiently high (pointing at resids value beyond histogram maximum at least):
        hist_argmax = full_hist.argmax()

        if hist_argmax >= findIndex(binz, resids[iter_index - 1], 0):
            raise ValueError(
                "Either the data histogram is too strangely shaped for oddWater OD detection - "
                "or a too low value for iter_start was passed (iter_start better be greater 0.5)"
            )
        # GO!
        iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)
        upper_tail_index = int(np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index))
        resids_tail_index = findIndex(resids, binz[upper_tail_index], 0)
        upper_tail_hist, bins = np.histogram(
            resids[resids_tail_index:iter_index], bins=binz[upper_tail_index : iter_max_bin_index + 1]
        )

        while (test_val < crit_val) & (iter_index < resids.size - 1):
            iter_index += 1
            new_iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)

            # following if/else block "manually" expands the data histogram and circumvents calculation of the complete
            # histogram in any new iteration.
            if new_iter_max_bin_index == iter_max_bin_index:
                upper_tail_hist[-1] += 1
            else:
                upper_tail_hist = np.append(upper_tail_hist, np.zeros([new_iter_max_bin_index - iter_max_bin_index]))
                upper_tail_hist[-1] += 1
                iter_max_bin_index = new_iter_max_bin_index
                upper_tail_index_new = int(np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index))
                upper_tail_hist = upper_tail_hist[upper_tail_index_new - upper_tail_index :]
                upper_tail_index = upper_tail_index_new

            # fitting
            lambdA, _ = curve_fit(
                fit_function,
                xdata=binzenters[upper_tail_index:iter_max_bin_index],
                ydata=upper_tail_hist,
                p0=[-np.log(alpha / resids[iter_index])],
            )

            crit_val = neg_log_alpha / lambdA
            test_val = resids[iter_index]

    # flag them!
    to_flag_index = val_frame.index[sorted_i[iter_index:]]
    for var in fields:
        flagger = flagger.setFlags(var, to_flag_index, **kwargs)

    return data, flagger


@register()
def spikes_flagRaise(
    data,
    field,
    flagger,
    thresh,
    raise_window,
    intended_freq,
    average_window=None,
    mean_raise_factor=2,
    min_slope=None,
    min_slope_weight=0.8,
    numba_boost=True,
    **kwargs,
):

    # NOTE1: this implementation accounts for the case of "pseudo" spikes that result from checking against outliers
    # NOTE2: the test is designed to work on raw data as well as on regularized
    #
    # See saqc documentation at:
    # https://git.ufz.de/rdm-software/saqc/blob/develop/docs/funcs/SpikeDetection.md
    # for more details

    # prepare input args
    dataseries = data[field].dropna()
    raise_window = pd.Timedelta(raise_window)
    intended_freq = pd.Timedelta(intended_freq)
    if min_slope is not None:
        min_slope = np.abs(min_slope)

    if average_window is None:
        average_window = 1.5 * pd.Timedelta(raise_window)

    if thresh < 0:
        dataseries *= -1
        thresh *= -1

    def raise_check(x, thresh):
        test_set = x[-1] - x[0:-1]
        max_val = np.max(test_set)
        if max_val >= thresh:
            return max_val
        else:
            return np.nan

    def custom_rolling_mean(x):
        return np.mean(x[:-1])

    # get invalid-raise/drop mask:
    raise_series = dataseries.rolling(raise_window, min_periods=2)

    if numba_boost:
        raise_check = numba.jit(raise_check, nopython=True)
        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True, engine="numba")
    else:
        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True)

    if raise_series.isna().all():
        return data, flagger

    # "unflag" values of unsifficient deviation to theire predecessors
    if min_slope is not None:
        w_mask = (
            pd.Series(dataseries.index).diff().dt.total_seconds() / intended_freq.total_seconds()
        ) > min_slope_weight
        slope_mask = np.abs(dataseries.diff()) < min_slope
        to_unflag = raise_series.notna() & w_mask.values & slope_mask
        raise_series[to_unflag] = np.nan

    # calculate and apply the weighted mean weights (pseudo-harmonization):
    weights = (
        pd.Series(dataseries.index).diff(periods=2).shift(-1).dt.total_seconds() / intended_freq.total_seconds() / 2
    )

    weights.iloc[0] = 0.5 + (dataseries.index[1] - dataseries.index[0]).total_seconds() / (
        intended_freq.total_seconds() * 2
    )

    weights.iloc[-1] = 0.5 + (dataseries.index[-1] - dataseries.index[-2]).total_seconds() / (
        intended_freq.total_seconds() * 2
    )

    weights[weights > 1.5] = 1.5
    weighted_data = dataseries.mul(weights.values)

    # rolling weighted mean calculation
    weighted_rolling_mean = weighted_data.rolling(average_window, min_periods=2, closed="both")
    if numba_boost:
        custom_rolling_mean = numba.jit(custom_rolling_mean, nopython=True)
        weighted_rolling_mean = weighted_rolling_mean.apply(custom_rolling_mean, raw=True, engine="numba")
    else:
        weighted_rolling_mean = weighted_rolling_mean.apply(custom_rolling_mean, raw=True)

    # check means against critical raise value:
    to_flag = dataseries >= weighted_rolling_mean + (raise_series / mean_raise_factor)
    to_flag &= raise_series.notna()
    flagger = flagger.setFlags(field, to_flag[to_flag].index, **kwargs)

    return data, flagger


@register()
def spikes_flagSlidingZscore(
    data, field, flagger, window, offset, count=1, polydeg=1, z=3.5, method="modZ", **kwargs,
):
    """ A outlier detection in a sliding window. The method for detection can be a simple Z-score or the more robust
    modified Z-score, as introduced here [1].

    The steps are:
    1.  a window of size `window` is cut from the data
    2.  the data is fit by a polynomial of the given degree `polydeg`
    3.  the outlier `method` detect potential outlier
    4.  the window is continued by `offset` to the next data-slot.
    5.  processing continue at 1. until end of data.
    6.  all potential outlier, that are detected `count`-many times, are promoted to real outlier and flagged by the `flagger`

    :param data:        pandas dataframe. holding the data
    :param field:       fieldname in `data`, which holds the relevant infos
    :param flagger:     flagger.
    :param window:      int or time-offset string (see [2]). The size of the window the outlier detection is run in. default: 1h
    :param offset:      int or time-offset string (see [2]). Stepsize the window is set further. default: 1h
    :param method:      str. `modZ`  or `zscore`. see [1] at section `Z-Scores and Modified Z-Scores`
    :param count:       int. this many times, a datapoint needs to be detected in different windows, to be finally
                        flagged as outlier
    :param polydeg:     The degree for the polynomial fit, to calculate the residuum
    :param z:           float. the value the (mod.) Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])

    Links:
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    [2] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    """

    use_offset = False
    dx_s = offset
    winsz_s = window
    # check param consistency
    if isinstance(window, str) or isinstance(offset, str):
        if isinstance(window, str) and isinstance(offset, str):
            use_offset = True
            dx_s = offset2seconds(offset)
            winsz_s = offset2seconds(window)
        else:
            raise TypeError(
                f"`window` and `offset` must both be an offset or both be numeric, {window} and {offset} was passed"
            )

    # check params
    if polydeg < 0:
        raise ValueError("polydeg must be positive")
    if z < 0:
        raise ValueError("z must be positive")
    if count <= 0:
        raise ValueError("count must be positive and not zero")

    if dx_s >= winsz_s and count == 1:
        pass
    elif dx_s >= winsz_s and count > 1:
        ValueError("If stepsize `offset` is bigger that the window-size, every value is seen just once, so use count=1")
    elif count > winsz_s // dx_s:
        raise ValueError(
            f"Adjust `offset`, `stepsize` or `window`. A single data point is "
            f"seen `floor(window / offset) = {winsz_s // dx_s}` times, but count is set to {count}"
        )

    # prepare the method
    method = method.lower()
    if method == "modz":

        def _calc(residual):
            diff = np.abs(residual - np.median(residual))
            mad = np.median(diff)
            return (mad > 0) & (0.6745 * diff > z * mad)

    elif method == "zscore":

        def _calc(residual):
            score = zscore(residual, ddof=1)
            return np.abs(score) > z

    else:
        raise NotImplementedError
    method = _calc

    # prepare data, work on numpy arrays for the fulfilling pleasure of performance
    d = data[field].dropna()
    all_indices = np.arange(len(d.index))
    x = (d.index - d.index[0]).total_seconds().values
    y = d.values
    counters = np.full(len(d.index), count)

    if use_offset:
        _loopfun = slidingWindowIndices
    else:

        def _loopfun(arr, wsz, step):
            for i in range(0, len(arr) - wsz + 1, step):
                yield i, i + wsz

    for start, end in _loopfun(d.index, window, offset):
        # mask points that have been already discarded
        mask = counters[start:end] > 0
        indices = all_indices[all_indices[start:end][mask]]
        xchunk = x[indices]
        ychunk = y[indices]

        if xchunk.size == 0:
            continue

        # get residual
        coef = poly.polyfit(xchunk, ychunk, polydeg)
        model = poly.polyval(xchunk, coef)
        residual = ychunk - model

        score = method(residual)

        # count`em in
        goneMad = score.nonzero()[0]
        counters[indices[goneMad]] -= 1

    outlier = np.where(counters <= 0)[0]
    loc = d[outlier].index
    flagger = flagger.setFlags(field, loc=loc, **kwargs)
    return data, flagger


@register()
def spikes_flagMad(data, field, flagger, window, z=3.5, **kwargs):
    """ The function represents an implementation of the modyfied Z-score outlier detection method, as introduced here:

    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    The test needs the input data to be harmonized to an equidustant time stamp series (=have frequencie))

    :param data:        The pandas dataframe holding the data-to-be flagged.
                        Data must be indexed by a datetime series and be harmonized onto a
                        time raster with seconds precision.
    :param field:       Fieldname of the Soil moisture measurements field in data.
    :param flagger:     A flagger - object. (saqc.flagger.X)
    :param winsz:      Offset String. Denoting the windows size that that th "Z-scored" values have to lie in.
    :param z:           Float. The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
    """
    d = data[field].copy().mask(flagger.isFlagged(field))
    median = d.rolling(window=window, closed="both").median()
    diff = (d - median).abs()
    mad = diff.rolling(window=window, closed="both").median()
    mask = (mad > 0) & (0.6745 * diff > z * mad)
    # NOTE:
    # In pandas <= 0.25.3, the window size is not fixed if the
    # window-argument to rolling is a frequency. That implies,
    # that during the first iterations the window has a size of
    # 1, 2, 3, ... until it eventually covers the disered time
    # span. For stuff the calculation of median, that is rather
    # unfortunate, as the size of calculation base might differ
    # heavily. So don't flag something until, the window reaches
    # its target size
    if not isinstance(window, int):
        index = mask.index
        mask.loc[index < index[0] + pd.to_timedelta(window)] = False

    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register()
def spikes_flagBasic(data, field, flagger, thresh=7, tolerance=0, window="15min", **kwargs):
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
    :param field:   String. Fieldname of the data column to be tested.
    :param flagger: saqc.flagger. A flagger - object.
    :param thresh:  Float. The lower bound for a value jump, to be considered as initialising a spike.
                    (see condition (1) in function description).
    :param tolerance: Float. Tolerance value.  (see condition (2) in function description)
    :param window_size:  Offset String. The time span in wich the values of a spikey course have to return to the normal
                    value course (see condition (3) in function description).
    :return:
    """

    dataseries = data[field].dropna()
    # get all the entries preceding a significant jump
    pre_jumps = dataseries.diff(periods=-1).abs() > thresh
    pre_jumps = pre_jumps[pre_jumps]
    if pre_jumps.empty:
        return data, flagger
    # get all the entries preceeding a significant jump and its successors within "length" range
    to_roll = pre_jumps.reindex(dataseries.index, method="ffill", tolerance=window, fill_value=False).dropna()

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
    to_roll = (
        to_roll.rolling(window, closed="both")
        .apply(spike_tester, args=(pre_jump_reversed_index, thresh, tolerance), raw=False)
        .astype(int)
    )
    # reconstruct original index and sequence
    to_roll = to_roll[::-1]
    to_roll.index = original_index
    to_write = to_roll[to_roll != 0]
    to_flag = pd.Index([])
    # here comes a loop...):
    for row in to_write.iteritems():
        loc = to_roll.index.get_loc(row[0])
        to_flag = to_flag.append(to_roll.iloc[loc + 1 : loc + row[1] + 1].index)

    to_flag = to_flag.drop_duplicates(keep="first")
    flagger = flagger.setFlags(field, to_flag, **kwargs)
    return data, flagger


@register()
def spikes_flagSpektrumBased(
    data,
    field,
    flagger,
    raise_factor=0.15,
    deriv_factor=0.2,
    noise_func="CoVar",
    noise_window="12h",
    noise_thresh=1,
    smooth_window=None,
    smooth_poly_deg=2,
    **kwargs,
):
    """
    This Function is a generalization of the Spectrum based Spike flagging mechanism as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    Function detects and flags spikes in input data series by evaluating its derivatives and applying some
    conditions to it. A datapoint is considered a spike, if:

    (1) the quotient to its preceeding datapoint exceeds a certain bound
    (controlled by param "raise_factor")
    (2) the quotient of the datas second derivate at the preceeding and subsequent timestamps is close enough to 1.
    (controlled by param "deriv_factor")
    (3) the surrounding data is not too noisy. (Coefficient of Variation[+/- noise_window] < 1)
    (controlled by param "noise_thresh")

    Some things you should be conscious about when applying this test:

       NOTE1: You should run less complex tests, especially range-tests, or absolute spike tests previously to this one,
       since the spike check for any potential, unflagged spike, is relatively costly
       (1 x smoothing + 2 x deviating + 2 x condition application).

       NOTE2: Due to inconsistency in the paper that provided the concept of this test [paper:], its not really clear
       weather to use the coefficient of variance or the relative variance for noise testing.
       Since the relative variance was explicitly denoted in the formulas, the function defaults to relative variance,
       but can be switched to coefficient of variance, by assignment to parameter "noise statistic".


       :param data:                 The pandas dataframe holding the data-to-be flagged.
                                    Data must be indexed by a datetime series and be harmonized onto a
                                    time raster with seconds precision.
       :param field:                Fieldname of the Soil moisture measurements field in data.
       :param flagger:              A flagger - object. (saqc.flagger.X)
       :param smooth_window:        Offset string. Size of the filter window, used to calculate the derivatives.
                                    (relevant only, if: diff_method='savgol')
       :param smooth_poly_deg:      Integer. Polynomial order, used for smoothing with savitzk golay filter.
                                    (relevant only, if: diff_method='savgol')
       :param raise_factor:         A float, determinating the bound, the quotient of two consecutive values
                                    has to exceed, to be regarded as potentially spike. A value of 0.x will
                                    trigger the spike test for value y_t, if:
                                    (y_t)/(y_t-1) > 1 + x or:
                                    (y_t)/(y_t-1) < 1 - x.
       :param deriv_factor:         A float, determining the interval, the quotient of the datas second derivate
                                    around a potential spike has to be part of, to trigger spike flagging for
                                    this value. A datapoint y_t will pass this spike condition if,
                                    for deriv_factor = 0.x, and the second derivative y'' of y, the condition:
                                    1 - x < abs((y''_t-1)/(y''_t+1)) < 1 + x
                                    holds
       :param noise_thresh:         A float, determining the bound, the data noisy-ness around a potential spike
                                    must not exceed, in order to guarantee a justifyed judgement:
                                    Therefor the coefficient selected by parameter noise_func (COVA),
                                    of all values within t +/- param "noise_window",
                                    but excluding the point y_t itself, is evaluated and tested
                                    for: COVA < noise_thresh.
       :param noise_window:         Offset string, determining the size of the window, the coefficient of
                                    variation is calculated of, to determine data noisy-ness around a potential
                                    spike.
                                    The potential spike y_t will be centered in a window of expansion:
                                    [y_t - noise_window_size, y_t + noise_window_size].
       :param noise_func:           String. Determines, wheather to use
                                    "relative variance" or "coefficient of variation" to check against the noise
                                    barrier.
                                    'CoVar' -> "Coefficient of variation"
                                    'rVar'  -> "relative Variance"
    """

    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)
    noise_func_map = {"covar": pd.Series.var, "rvar": pd.Series.std}
    noise_func = noise_func_map[noise_func.lower()]

    if smooth_window is None:
        smooth_window = 3 * pd.Timedelta(data_rate)
    else:
        smooth_window = pd.Timedelta(smooth_window)

    quotient_series = dataseries / dataseries.shift(+1)
    spikes = (quotient_series > (1 + raise_factor)) | (quotient_series < (1 - raise_factor))
    spikes = spikes[spikes == True]

    # loop through spikes: (loop may sound ugly - but since the number of spikes is supposed to not exceed the
    # thousands for year data - a loop going through all the spikes instances is much faster than
    # a rolling window, rolling all through a stacked year dataframe )

    # calculate some values, repeatedly needed in the course of the loop:

    filter_window_seconds = smooth_window.seconds
    smoothing_periods = int(np.ceil((filter_window_seconds / data_rate.n)))
    lower_dev_bound = 1 - deriv_factor
    upper_dev_bound = 1 + deriv_factor

    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for spike in spikes.index:
        start_slice = spike - smooth_window
        end_slice = spike + smooth_window

        scnd_derivate = savgol_filter(
            dataseries[start_slice:end_slice], window_length=smoothing_periods, polyorder=smooth_poly_deg, deriv=2,
        )

        length = scnd_derivate.size
        test_ratio_1 = np.abs(scnd_derivate[int(((length + 1) / 2) - 2)] / scnd_derivate[int(((length + 1) / 2))])

        if lower_dev_bound < test_ratio_1 < upper_dev_bound:
            # apply noise condition:
            start_slice = spike - pd.Timedelta(noise_window)
            end_slice = spike + pd.Timedelta(noise_window)
            test_slice = dataseries[start_slice:end_slice].drop(spike)
            test_ratio_2 = np.abs(noise_func(test_slice) / test_slice.mean())
            # not a spike, we want to flag, if condition not satisfied:
            if test_ratio_2 > noise_thresh:
                spikes[spike] = False

        # not a spike, we want to flag, if condition not satisfied
        else:
            spikes[spike] = False

    spikes = spikes[spikes == True]

    flagger = flagger.setFlags(field, spikes.index, **kwargs)
    return data, flagger
