#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
The module gathers all kinds of timeseries tranformations.
"""
import re
import warnings
from typing import Union
import sys

import pandas as pd
import numpy as np
import numba as nb
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr, median_abs_deviation
from scipy.signal import filtfilt, butter
import numpy.polynomial.polynomial as poly


def identity(ts):
    # identity function
    return ts


def count(ts):
    # count is a dummy to trigger according built in count method of resamplers when
    # passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.count()


def first(ts):
    # first is a dummy to trigger according built in count method of resamplers when
    # passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.first()


def last(ts):
    # last is a dummy to trigger according built in count method of resamplers when
    # passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.last()


def zeroLog(ts):
    # zero log returns np.nan instead of -np.inf, when passed 0. Usefull, because
    # in internal processing, you only have to check for nan values if you need to
    # remove "invalidish" values from the data.
    log_ts = np.log(ts)
    log_ts[log_ts == -np.inf] = sys.float_info.min
    return log_ts


def derivative(ts, unit="1min"):
    # calculates derivative of timeseries, expressed in slope per "unit"
    return ts / (deltaT(ts, unit=unit))


def deltaT(ts, unit="1min"):
    # calculates series of time gaps in ts
    return (
        ts.index.to_series().diff().dt.total_seconds()
        / pd.Timedelta(unit).total_seconds()
    )


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
    # standardization with median and inter quantile range
    return (ts - np.median(ts)) / iqr(ts, nan_policy="omit")


def kNN(in_arr, n_neighbors, algorithm="ball_tree", metric="minkowski", p=2):
    # k-nearest-neighbor search

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm, metric=metric, p=p
    ).fit(in_arr.reshape(in_arr.shape[0], -1))
    return nbrs.kneighbors()


def maxGap(in_arr):
    """
    Search for the maximum gap in an array of sorted distances (func for scoring kNN
    distance matrice)
    """
    return max(in_arr[0], max(np.diff(in_arr)))


@nb.njit
def _maxConsecutiveNan(arr, max_consec):
    # checks if arr (boolean array) has not more then "max_consec" consecutive True
    # values
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
    # data has to be boolean. False=Valid Value, True=invalid Value function returns
    # True-array of input array size for invalid input arrays False array for valid
    # ones
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
    return np.nanstd(
        data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)], ddof=1
    )


def varQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    return np.nanvar(
        data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)], ddof=1
    )


def meanQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    return np.nanmean(
        data[~validationTrafo(data.isna(), max_nan_total, max_nan_consec)]
    )


def interpolateNANs(
    data, method, order=2, inter_limit=2, downgrade_interpolation=False
):
    """
    The function interpolates nan-values (and nan-grids) in timeseries data. It can
    be passed all the method keywords from the pd.Series.interpolate method and will
    than apply this very methods. Note, that the limit keyword really restricts
    the interpolation to chunks, not containing more than "limit" nan entries (
    thereby not being identical to the "limit" keyword of pd.Series.interpolate).

    :param data:                    pd.Series or np.array. The data series to be interpolated
    :param method:                  String. Method keyword designating interpolation method to use.
    :param order:                   Integer. If your desired interpolation method needs an order to be passed -
                                    here you pass it.
    :param inter_limit:             Integer. Default = 2. Limit up to which consecutive nan - values in the data get
                                    replaced by interpolation.
                                    Its default value suits an interpolation that only will apply to points of an
                                    inserted frequency grid. (regularization by interpolation)
                                    Gaps wider than "limit" will NOT be interpolated at all.
    :param downgrade_interpolation:  Boolean. Default False. If True:
                                    If a data chunk not contains enough values for interpolation of the order "order",
                                    the highest order possible will be selected for that chunks interpolation.

    :return:
    """
    inter_limit = int(inter_limit)
    data = pd.Series(data, copy=True)
    gap_mask = data.isna().rolling(inter_limit, min_periods=0).sum() != inter_limit

    if inter_limit == 2:
        gap_mask = gap_mask & gap_mask.shift(-1, fill_value=True)
    else:
        gap_mask = (
            gap_mask.replace(True, np.nan)
            .fillna(method="bfill", limit=inter_limit)
            .replace(np.nan, True)
            .astype(bool)
        )

    pre_index = data.index
    data = data[gap_mask]

    if method in ["linear", "time"]:

        data.interpolate(
            method=method, inplace=True, limit=inter_limit - 1, limit_area="inside"
        )

    else:
        dat_name = data.name
        gap_mask = (~gap_mask).cumsum()
        data = pd.merge(gap_mask, data, how="inner", left_index=True, right_index=True)

        def _interpolWrapper(x, wrap_order=order, wrap_method=method):
            if x.count() > wrap_order:
                try:
                    return x.interpolate(method=wrap_method, order=int(wrap_order))
                except (NotImplementedError, ValueError):
                    warnings.warn(
                        f"Interpolation with method {method} is not supported at order "
                        f"{wrap_order}. and will be performed at order {wrap_order-1}"
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
        # squeezing the 1-dimensional frame resulting from groupby for consistency
        # reasons
        data = data.squeeze(axis=1)
        data.name = dat_name
    data = data.reindex(pre_index)

    return data


def aggregate2Freq(
    data: pd.Series,
    method,
    freq,
    agg_func,
    fill_value=np.nan,
    max_invalid_total=None,
    max_invalid_consec=None,
):
    """
    The function aggregates values to an equidistant frequency grid with func.
    Timestamps that gets no values projected, get filled with the fill-value. It
    also serves as a replacement for "invalid" intervals.
    """
    methods = {
        # offset, closed, label
        "nagg": lambda f: (f / 2, "left", "left"),
        "bagg": lambda _: (pd.Timedelta(0), "left", "left"),
        "fagg": lambda _: (pd.Timedelta(0), "right", "right"),
    }

    # filter data for invalid patterns (since filtering is expensive we pre-check if
    # it is demanded)
    if max_invalid_total is not None or max_invalid_consec is not None:
        if pd.isna(fill_value):
            temp_mask = data.isna()
        else:
            temp_mask = data == fill_value

        temp_mask = temp_mask.groupby(pd.Grouper(freq=freq)).transform(
            validationTrafo,
            max_nan_total=max_invalid_total,
            max_nan_consec=max_invalid_consec,
        )
        data[temp_mask] = fill_value

    freq = pd.Timedelta(freq)
    offset, closed, label = methods[method](freq)

    resampler = data.resample(
        freq, closed=closed, label=label, origin="start_day", offset=offset
    )

    # count valid values
    counts = resampler.count()

    # native methods of resampling (median, mean, sum, ..) are much faster than apply
    try:
        check_name = re.sub("^nan", "", agg_func.__name__)
        # a nasty special case: if function "count" was passed, we not want empty
        # intervals to be replaced by fill_value:
        if check_name == "count":
            data = counts.copy()
            counts[:] = np.nan
        else:
            data = getattr(resampler, check_name)()
    except AttributeError:
        data = resampler.apply(agg_func)

    # we custom fill bins that have no value
    data[counts == 0] = fill_value

    # undo the temporary shift, to mimic centering the frequency
    if method == "nagg":
        data.index += offset

    return data


def shift2Freq(
    data: Union[pd.Series, pd.DataFrame], method: str, freq: str, fill_value
):
    """
    shift timestamps backwards/forwards in order to align them with an equidistant
    frequency grid. Resulting Nan's are replaced with the fill-value.
    """

    methods = {
        "fshift": lambda freq: ("ffill", pd.Timedelta(freq)),
        "bshift": lambda freq: ("bfill", pd.Timedelta(freq)),
        "nshift": lambda freq: ("nearest", pd.Timedelta(freq) / 2),
    }
    direction, tolerance = methods[method](freq)
    target_ind = pd.date_range(
        start=data.index[0].floor(freq),
        end=data.index[-1].ceil(freq),
        freq=freq,
        name=data.index.name,
    )
    return data.reindex(
        target_ind, method=direction, tolerance=tolerance, fill_value=fill_value
    )


def butterFilter(
    x, cutoff, nyq=0.5, filter_order=2, fill_method="linear", filter_type="low"
):
    """
    Applies butterworth filter.
    `x` is expected to be regularly sampled.

    Parameters
    ----------
    x: pd.Series
        input timeseries

    cutoff: float
        The cutoff-frequency, expressed in multiples of the sampling rate.

    nyq: float
        The niquist-frequency. expressed in multiples if the sampling rate.

    fill_method: Literal[‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’]
        Fill method to be applied on the data before filtering (butterfilter cant
        handle ''np.nan''). See documentation of pandas.Series.interpolate method for
        details on the methods associated with the different keywords.


    Returns
    -------
    """
    na_mask = x.isna()
    x = x.interpolate(fill_method).interpolate("ffill").interpolate("bfill")
    b, a = butter(N=filter_order, Wn=cutoff / nyq, btype=filter_type)
    y = pd.Series(filtfilt(b, a, x), x.index, name=x.name)
    y[na_mask] = np.nan
    return y


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
    # a function to roll with, for polynomial fitting of data not having an
    # equidistant frequency grid. (expects to get passed pandas timeseries),
    # so raw parameter of rolling.apply should be set to False.
    x_data = ((in_slice.index - in_slice.index[0]).total_seconds()) / 60
    fitted = poly.polyfit(x_data, in_slice.values, poly_deg)
    center_pos = int(len(in_slice) - center_index_ser[in_slice.index[-1]])
    return poly.polyval(x_data[center_pos], fitted)


def expModelFunc(x, a=0, b=0, c=0):
    # exponential model function, used in optimization contexts (drift correction)
    return a + b * (np.exp(c * x) - 1)


def expDriftModel(x, c, origin, target):
    c = abs(c)
    b = (target - origin) / (np.exp(c) - 1)
    return expModelFunc(x, origin, b, c)


def linearDriftModel(x, origin, target):
    return origin + x * target


def linearInterpolation(data, inter_limit=2):
    return interpolateNANs(data, "time", inter_limit=inter_limit)


def polynomialInterpolation(data, inter_limit=2, inter_order=2):
    return interpolateNANs(
        data, "polynomial", inter_limit=inter_limit, order=inter_order
    )
