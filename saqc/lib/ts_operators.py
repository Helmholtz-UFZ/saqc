#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
The module gathers all kinds of timeseries tranformations.
"""
import logging

import re

import pandas as pd
import numpy as np
import numba as nb

from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr, median_abs_deviation
import numpy.polynomial.polynomial as poly

logger = logging.getLogger("SaQC")


def identity(ts):
    # identity function
    return ts


def count(ts):
    # count is a dummy to trigger according built in count method of
    # resamplers when passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.count()


def first(ts):
    # first is a dummy to trigger according built in count method of
    # resamplers when passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.first()


def last(ts):
    # last is a dummy to trigger according built in count method of
    # resamplers when passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.last()


def zeroLog(ts):
    # zero log returns np.nan instead of -np.inf, when passed 0. Usefull, because
    # in internal processing, you only have to check for nan values if you need to
    # remove "invalidish" values from the data.
    log_ts = np.log(ts)
    log_ts[log_ts == -np.inf] = np.nan
    return log_ts


def derivative(ts, unit="1min"):
    # calculates derivative of timeseries, expressed in slope per "unit"
    return ts / (deltaT(ts, unit=unit))


def deltaT(ts, unit="1min"):
    # calculates series of time gaps in ts
    return ts.index.to_series().diff().dt.total_seconds() / pd.Timedelta(unit).total_seconds()


def difference(ts):
    # NOTE: index of input series gets lost!
    return np.diff(ts, prepend=np.nan)


def rateOfChange(ts):
    return difference(ts) / ts


def relativeDifference(ts):
    res = ts - 0.5 * (np.roll(ts, +1) + np.roll(ts, -1))
    res[0] = np.nan
    res[-1] = np.nan
    return res


def scale(ts, target_range=1, projection_point=None):
    # scales input series to have values ranging from - target_rang to + target_range
    if not projection_point:
        projection_point = np.max(np.abs(ts))
    return (ts / projection_point) * target_range


def normScale(ts):
    # func scales series to [0,1] interval and projects constant series onto 0.5
    ts_min = ts.min()
    ts_max = ts.max()
    if ts_min == ts_max:
        return pd.Series(data=0.5, index=ts.index)
    else:
        return (ts - ts_min) / (ts.max() - ts_min)


def standardizeByMean(ts):
    # standardization with mean and probe variance
    return (ts - np.mean(ts)) / np.std(ts, ddof=1)


def standardizeByMedian(ts):
    # standardization with median (MAD)
    # NO SCALING
    return (ts - np.median(ts)) / median_abs_deviation(ts, nan_policy="omit")


def standardizeByIQR(ts):
    # standardization with median and interquartile range
    return (ts - np.median(ts)) / iqr(ts, nan_policy="omit")


def kNN(in_arr, n_neighbors, algorithm="ball_tree", metric='minkowski', p=2, radius=None):
    # k-nearest-neighbor search

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric, p=p)\
        .fit(in_arr.reshape(in_arr.shape[0], -1))
    if radius is None:
        return nbrs.kneighbors()

    rad_nbrs = nbrs.radius_neighbors(radius=radius)
    dist = np.zeros((in_arr.shape[0], n_neighbors))
    dist[:] = np.nan
    i = 0
    for k in rad_nbrs[0]:
        dist[i, 0:len(k)] = k
        i += 1
    return dist, np.array([])


def maxGap(in_arr):
    """
    Search for the maximum gap in an array of sorted distances (func for scoring kNN distance matrice)
    """
    return max(in_arr[0], max(np.diff(in_arr)))


@nb.njit
def _maxConsecutiveNan(arr, max_consec):
    # checks if arr (boolean array) has not more then "max_consec" consecutive True values
    current = 0
    idx = 0
    while idx < arr.size:
        while idx < arr.size and arr[idx]:
            current += 1
            idx += 1
        if current > max_consec:
            return False
        current = 0
        idx += 1
    return True


def validationTrafo(data, max_nan_total, max_nan_consec):
    # data has to be boolean. False=Valid Value, True=invalid Value
    # function returns True-array of input array size for invalid input arrays False array for valid ones
    data = data.copy()
    if (max_nan_total is np.inf) & (max_nan_consec is np.inf):
        return data

    # nan_mask = np.isnan(data)

    if data.sum() <= max_nan_total:
        if max_nan_consec is np.inf:
            data[:] = False
            return data
        elif _maxConsecutiveNan(np.asarray(data), max_nan_consec):
            data[:] = False
        else:
            data[:] = True
    else:
        data[:] = True

    return data


def stdQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    return np.nanstd(data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)], ddof=1)


def varQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    return np.nanvar(data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)], ddof=1)


def meanQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    return np.nanmean(data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)])


def interpolateNANs(data, method, order=2, inter_limit=2, downgrade_interpolation=False, return_chunk_bounds=False):
    """
    The function interpolates nan-values (and nan-grids) in timeseries data. It can be passed all the method keywords
    from the pd.Series.interpolate method and will than apply this very methods. Note, that the inter_limit keyword
    really restricts the interpolation to chunks, not containing more than "inter_limit" nan entries
    (thereby not being identical to the "limit" keyword of pd.Series.interpolate).

    :param data:                    pd.Series or np.array. The data series to be interpolated
    :param method:                  String. Method keyword designating interpolation method to use.
    :param order:                   Integer. If your desired interpolation method needs an order to be passed -
                                    here you pass it.
    :param inter_limit:             Integer. Default = 2. Limit up to which consecutive nan - values in the data get
                                    replaced by interpolation.
                                    Its default value suits an interpolation that only will apply to points of an
                                    inserted frequency grid. (regularization by interpolation)
                                    Gaps wider than "inter_limit" will NOT be interpolated at all.
    :param downgrade_interpolation:  Boolean. Default False. If True:
                                    If a data chunk not contains enough values for interpolation of the order "order",
                                    the highest order possible will be selected for that chunks interpolation.
    :param return_chunk_bounds:     Boolean. Default False. If True:
                                    Additionally to the interpolated data, the start and ending points of data chunks
                                    not containing no series consisting of more then "inter_limit" nan values,
                                    are calculated and returned.
                                    (This option fits requirements of the "interpolateNANs" functions use in the
                                    context of saqc harmonization mainly.)

    :return:
    """
    inter_limit = int(inter_limit)
    data = pd.Series(data).copy()
    gap_mask = (data.rolling(inter_limit, min_periods=0).apply(lambda x: np.sum(np.isnan(x)), raw=True)) != inter_limit

    if inter_limit == 2:
        gap_mask = gap_mask & gap_mask.shift(-1, fill_value=True)
    else:
        gap_mask = (
            gap_mask.replace(True, np.nan).fillna(method="bfill", limit=inter_limit).replace(np.nan, True).astype(bool)
        )

    if return_chunk_bounds:
        # start end ending points of interpolation chunks have to be memorized to block their flagging:
        chunk_switches = gap_mask.astype(int).diff()
        chunk_starts = chunk_switches[chunk_switches == -1].index
        chunk_ends = chunk_switches[(chunk_switches.shift(-1) == 1)].index
        chunk_bounds = chunk_starts.join(chunk_ends, how="outer", sort=True)

    pre_index = data.index
    data = data[gap_mask]

    if method in ["linear", "time"]:

        data.interpolate(method=method, inplace=True, limit=inter_limit - 1, limit_area="inside")

    else:
        dat_name = data.name
        gap_mask = (~gap_mask).cumsum()
        data = pd.merge(gap_mask, data, how="inner", left_index=True, right_index=True)

        def _interpolWrapper(x, wrap_order=order, wrap_method=method):
            if x.count() > wrap_order:
                try:
                    return x.interpolate(method=wrap_method, order=int(wrap_order))
                except (NotImplementedError, ValueError):
                    logger.warning(
                        f"Interpolation with method {method} is not supported at order {wrap_order}. "
                        f"and will be performed at order {wrap_order-1}"
                    )
                    return _interpolWrapper(x, int(wrap_order - 1), wrap_method)
            elif x.size < 3:
                return x
            else:
                if downgrade_interpolation:
                    return _interpolWrapper(x, int(x.count() - 1), wrap_method)
                else:
                    return x

        data = data.groupby(data.columns[0]).transform(_interpolWrapper)
        # squeezing the 1-dimensional frame resulting from groupby for consistency reasons
        data = data.squeeze(axis=1)
        data.name = dat_name
    data = data.reindex(pre_index)
    if return_chunk_bounds:
        return data, chunk_bounds
    else: return data


def aggregate2Freq(
    data, method, freq, agg_func, fill_value=np.nan, max_invalid_total=None, max_invalid_consec=None
):
    # The function aggregates values to an equidistant frequency grid with agg_func.
    # Timestamps that have no values projected on them, get "fill_value" assigned. Also,
    # "fill_value" serves as replacement for "invalid" intervals

    methods = {
        "nagg": lambda seconds_total: (seconds_total/2, "left", "left"),
        "bagg": lambda _: (0, "left", "left"),
        "fagg": lambda _: (0, "right", "right"),
    }

    # filter data for invalid patterns (since filtering is expensive we pre-check if it is demanded)
    if (max_invalid_total is not None) | (max_invalid_consec is not None):
        if pd.isnull(fill_value):
            temp_mask = data.isna()
        else:
            temp_mask = data == fill_value

        temp_mask = temp_mask.groupby(pd.Grouper(freq=freq)).transform(
            validationTrafo, max_nan_total=max_invalid_total, max_nan_consec=max_invalid_consec
        )
        data[temp_mask] = fill_value

    seconds_total = pd.Timedelta(freq).total_seconds()
    base, label, closed = methods[method](seconds_total)

    # In the following, we check for empty intervals outside resample.apply, because:
    # - resample AND groupBy do insert value zero for empty intervals if resampling with any kind of "sum" application -
    #   we want "fill_value" to be inserted
    # - we are aggregating data and flags with this function and empty intervals usually would get assigned BAD
    #   flag (where resample inserts np.nan or 0)

    data_resampler = data.resample(f"{seconds_total:.0f}s", base=base, closed=closed, label=label)

    empty_intervals = data_resampler.count() == 0
    # great performance gain can be achieved, when avoiding .apply and using pd.resampler
    # methods instead. (this covers all the basic func aggregations, such as median, mean, sum, count, ...)
    try:
        check_name = re.sub("^nan", "", agg_func.__name__)
        # a nasty special case: if function "count" was passed, we not want empty intervals to be replaced by nan:
        if check_name == 'count':
            empty_intervals[:] = False
        data = getattr(data_resampler, check_name)()
    except AttributeError:
        data = data_resampler.apply(agg_func)

    # since loffset keyword of pandas.resample "discharges" after one use of the resampler (pandas logic) - we correct the
    # resampled labels offset manually, if necessary.
    if method == "nagg":
        data.index = data.index.shift(freq=pd.Timedelta(freq) / 2)
        empty_intervals.index = empty_intervals.index.shift(freq=pd.Timedelta(freq) / 2)
    data[empty_intervals] = fill_value

    return data


def shift2Freq(data, method, freq, fill_value=np.nan):
    # shift timestamps backwards/forwards in order to allign them with an equidistant
    # frequencie grid.

    methods = {
        "fshift": lambda freq: ("ffill", pd.Timedelta(freq)),
        "bshift": lambda freq: ("bfill", pd.Timedelta(freq)),
        "nshift": lambda freq: ("nearest", pd.Timedelta(freq)/2),
    }
    direction, tolerance = methods[method](freq)
    target_ind = pd.date_range(
        start=data.index[0].floor(freq), end=data.index[-1].ceil(freq),
        freq=freq,
        name=data.index.name
    )
    return data.reindex(target_ind, method=direction, tolerance=tolerance, fill_value=fill_value)


@nb.njit
def _coeffMat(x, deg):
    # helper function to construct numba-compatible polynomial fit function
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x ** n
    return mat_


@nb.jit
def _fitX(a, b):
    # helper function to construct numba-compatible polynomial fit function
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@nb.jit
def _fitPoly(x, y, deg):
    # a numba compatible polynomial fit function
    a = _coeffMat(x, deg)
    p = _fitX(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@nb.jit
def evalPolynomial(P, x):
    # a numba compatible polynomial evaluator
    result = 0
    for coeff in P:
        result = x * result + coeff
    return result


def polyRollerNumba(in_slice, miss_marker, val_range, center_index, poly_deg):
    # numba compatible function to roll with when modelling data with polynomial model
    miss_mask = in_slice == miss_marker
    x_data = val_range[~miss_mask]
    y_data = in_slice[~miss_mask]
    fitted = _fitPoly(x_data, y_data, deg=poly_deg)
    return evalPolynomial(fitted, center_index)


def polyRollerNoMissingNumba(in_slice, val_range, center_index, poly_deg):
    # numba compatible function to roll with when modelling data with polynomial model -
    # it is assumed, that in slice is an equidistant sample
    fitted = _fitPoly(val_range, in_slice, deg=poly_deg)
    return evalPolynomial(fitted, center_index)


def polyRoller(in_slice, miss_marker, val_range, center_index, poly_deg):
    # function to roll with when modelling data with polynomial model
    miss_mask = in_slice == miss_marker
    x_data = val_range[~miss_mask]
    y_data = in_slice[~miss_mask]
    fitted = poly.polyfit(x=x_data, y=y_data, deg=poly_deg)
    return poly.polyval(center_index, fitted)


def polyRollerNoMissing(in_slice, val_range, center_index, poly_deg):
    # function to roll with when modelling data with polynomial model
    # it is assumed, that in slice is an equidistant sample
    fitted = poly.polyfit(x=val_range, y=in_slice, deg=poly_deg)
    return poly.polyval(center_index, fitted)


def polyRollerIrregular(in_slice, center_index_ser, poly_deg):
    # a function to roll with, for polynomial fitting of data not having an equidistant frequency grid.
    # (expects to get passed pandas timeseries), so raw parameter of rolling.apply should be set to False.
    x_data = ((in_slice.index - in_slice.index[0]).total_seconds()) / 60
    fitted = poly.polyfit(x_data, in_slice.values, poly_deg)
    center_pos = int(len(in_slice) - center_index_ser[in_slice.index[-1]])
    return poly.polyval(x_data[center_pos], fitted)


def expModelFunc(x, a=0, b=0, c=0):
    # exponential model function, used in optimization contexts (drift correction)
    return a + b * (np.exp(c * x) - 1)


def linearInterpolation(data, inter_limit=2):
    return interpolateNANs(data, "time", inter_limit=inter_limit)


def polynomialInterpolation(data, inter_limit=2, inter_order=2):
    return interpolateNANs(data, "polynomial", inter_limit=inter_limit, order=inter_order)
