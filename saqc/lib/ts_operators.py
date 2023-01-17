#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

"""
The module gathers all kinds of timeseries tranformations.
"""
import re
import sys
import warnings
from typing import Union

import numba as nb
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import iqr, median_abs_deviation
from sklearn.neighbors import NearestNeighbors

from saqc.lib.tools import getFreqDelta


def identity(ts):
    """
    Returns the input.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    Returns
    -------
    ts: pd.Series
        the original
    """
    return ts


def count(ts):
    # count is a dummy to trigger according built in count method of resamplers when
    # passed to aggregate2freq. For consistency reasons, it works accordingly when
    # applied directly:
    return ts.count()


def zeroLog(ts):
    """
    Calculate log of values of series for (0, inf] and NaN otherwise.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    Returns
    -------
    pd.Series
    """
    log_ts = np.log(ts)
    log_ts[log_ts == -np.inf] = sys.float_info.min
    return log_ts


def derivative(ts, unit="1min"):
    """
    Calculates derivative of timeseries, expressed in slope per `unit`.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    unit : str
        Datetime offset unit.

    Returns
    -------
    pd.Series
    """
    return ts / deltaT(ts, unit=unit)


def deltaT(ts, unit="1min"):
    """
    Calculate the time difference of the index-values in seconds.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    Returns
    -------
    pd.Series
    """
    return (
        ts.index.to_series().diff().dt.total_seconds()
        / pd.Timedelta(unit).total_seconds()
    )


def difference(ts):
    """
    Calculate the difference of subsequent values in the series.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    Returns
    -------
    pd.Series
    """
    return ts.diff(1)


def rateOfChange(ts):
    """
    Calculate the rate of change of the series values.

    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.

    Returns
    -------
    pd.Series
    """
    return difference(ts) / ts


def relativeDifference(ts):
    res = ts - 0.5 * (np.roll(ts, +1) + np.roll(ts, -1))
    res[0] = np.nan
    res[-1] = np.nan
    return res


def scale(ts, target_range=1, projection_point=None):
    """
    Scales input series values to a given range.


    Parameters
    ----------
    ts : pd.Series
        A series with datetime index.
    target_range : int
        The projection will range from ``[-target_range, target_range]``

    Returns
    -------
    scaled: pd.Series
        The scaled Series
    """
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
def _exceedConsecutiveNanLimit(arr, max_consec):
    """
    Check if array has more consecutive NaNs than allowed.

    Parameters
    ----------
    arr : bool array
        boolean array
    max_consec : int
        maximum allowed consecutive `True`s

    Returns
    -------
    exceeded: bool
        True if more than allowed consecutive NaNs appear, False otherwise.
    """
    current = 0
    idx = 0
    while idx < arr.size:
        while idx < arr.size and arr[idx]:
            current += 1
            idx += 1
        if current > max_consec:
            return True
        current = 0
        idx += 1
    return False


def validationTrafo(data, max_nan_total, max_nan_consec):
    # data has to be boolean. False=Valid Value, True=invalid Value function returns
    # True-array of input array size for invalid input arrays False array for valid
    # ones
    data = data.copy()
    if max_nan_total is np.inf and max_nan_consec is np.inf:
        return data

    if data.sum() > max_nan_total:
        value = True
    elif max_nan_consec is np.inf:
        value = False
    else:
        value = _exceedConsecutiveNanLimit(np.asarray(data), max_nan_consec)

    data[:] = value
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


def _interpolWrapper(
    x, order=1, method="time", limit_area="inside", limit_direction=None
):
    """
    Function that automatically modifies the interpolation level or returns uninterpolated
    input data if the data configuration breaks the interpolation method at the selected degree.
    """

    min_vals_dict = {
        "nearest": 2,
        "slinear": 2,
        "quadratic": 3,
        "cubic": 4,
        "spline": order + 1,
        "polynomial": order + 1,
        "piecewise_polynomial": 2,
        "pchip": 2,
        "akima": 2,
        "cubicspline": 2,
    }
    min_vals = min_vals_dict.get(method, 0)

    if (x.size < 3) | (x.count() < min_vals):
        return x
    else:
        return x.interpolate(
            method=method,
            order=order,
            limit_area=limit_area,
            limit_direction=limit_direction,
        )


def interpolateNANs(data, method, order=2, gap_limit=2, extrapolate=None):
    """
    The function interpolates nan-values (and nan-grids) in timeseries data. It can
    be passed all the method keywords from the pd.Series.interpolate method and will
    than apply this very methods. Note, that the limit keyword really restricts
    the interpolation to gaps, not containing more than "limit" nan entries (
    thereby not being identical to the "limit" keyword of pd.Series.interpolate).

    :param data:                    pd.Series or np.array. The data series to be interpolated
    :param method:                  String. Method keyword designating interpolation method to use.
    :param order:                   Integer. If your desired interpolation method needs an order to be passed -
                                    here you pass it.
    :param gap_limit:               Integer or Offset String. Default = 2.
                                    Number up to which consecutive nan - values in the data get
                                    replaced by interpolated values.
                                    Its default value suits an interpolation that only will apply to points of an
                                    inserted frequency grid. (regularization by interpolation)
                                    Gaps of size "limit" or greater will NOT be interpolated at all.
    :param extrapolate:             Str or None. Default None. If True:
                                    If a data chunk not contains enough values for interpolation of the order "order",
                                    the highest order possible will be selected for that chunks interpolation.

    :return:
    """

    # helper variable for checking numerical value of gap limit, if its a numeric value (to avoid comparison to str)
    gap_check = np.nan if isinstance(gap_limit, str) else gap_limit
    data = pd.Series(data, copy=True)
    limit_area = "inside" if not extrapolate else "outside"
    if gap_check is None:
        # if there is actually no limit set to the gaps to-be interpolated, generate a dummy mask for the gaps
        gap_mask = pd.Series(True, index=data.index, name=data.name)
    else:
        if gap_check < 2:
            # breaks execution down the line and is thus catched here since it basically means "do nothing"
            return data
        else:
            # if there is a limit to the gaps to be interpolated, generate a mask that evaluates to False at the right
            # side of each too-large gap with a rolling.sum combo
            gap_mask = data.rolling(gap_limit, min_periods=0).count() > 0

            # correction for initial gap
            if isinstance(gap_limit, int):
                gap_mask.iloc[:gap_limit] = True

            if gap_limit == 2:
                # for the common case of gap_limit=2 (default "harmonisation"), we efficiently back propagate the False
                # value to fill the whole too-large gap by a shift and a conjunction.
                gap_mask = gap_mask & gap_mask.shift(-1, fill_value=True)
            else:
                # If the gap_size is bigger we make a flip-rolling combo to backpropagate the False values
                gap_mask = ~(
                    (~gap_mask[::-1]).rolling(gap_limit, min_periods=0).sum() > 0
                )[::-1]

    # memorizing the index for later reindexing
    pre_index = data.index
    # drop the gaps that are too large with regard to the gap_limit from the data-to-be interpolated
    data = data[gap_mask]
    if data.empty:
        return data

    if method in ["linear", "time"]:
        # in the case of linear interpolation, not much can go wrong/break so this conditional branch has efficient
        # finish by just calling pandas interpolation routine to fill the gaps remaining in the data:
        data.interpolate(
            method=method,
            inplace=True,
            limit_area=limit_area,
            limit_direction=extrapolate,
        )

    else:
        # if the method that is interpolated with, depends on not only the left and right border points of any gap,
        # but includes more points, it has to be applied on any data chunk seperated by the too-big gaps individually.
        # So we use the gap_mask to group the data into chunks and perform the interpolation on every chunk seperatly
        # with the .transform method of the grouper.
        gap_mask = (~gap_mask).cumsum()[data.index]
        chunk_groups = data.groupby(by=gap_mask)
        data = chunk_groups.transform(
            _interpolWrapper,
            **{
                "order": order,
                "method": method,
                "limit_area": limit_area,
                "limit_direction": extrapolate,
            },
        )
    # finally reinsert the dropped data gaps
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
        start=pd.Timestamp(data.index[0]).floor(freq),
        end=pd.Timestamp(data.index[-1]).ceil(freq),
        freq=freq,
        name=data.index.name,
    )
    return data.reindex(
        target_ind, method=direction, tolerance=tolerance, fill_value=fill_value
    )


def butterFilter(
    x, cutoff, nyq=0.5, filter_order=2, fill_method="linear", filter_type="lowpass"
):
    """
    Applies butterworth filter.
    `x` is expected to be regularly sampled.

    Parameters
    ----------
    x: pd.Series
        input timeseries

    cutoff: {float, str}
        The cutoff-frequency, either an offset freq string, or expressed in multiples of the sampling rate.

    nyq: float
        The niquist-frequency. expressed in multiples if the sampling rate.

    fill_method: Literal[‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’]
        Fill method to be applied on the data before filtering (butterfilter cant
        handle ''np.nan''). See documentation of pandas.Series.interpolate method for
        details on the methods associated with the different keywords.

    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
        The type of filter. Default is ‘lowpass’.

    Returns
    -------
    """
    if isinstance(cutoff, str):
        cutoff = getFreqDelta(x.index) / pd.Timedelta(cutoff)

    na_mask = x.isna()
    x = x.interpolate(fill_method).interpolate("ffill").interpolate("bfill")
    b, a = butter(N=filter_order, Wn=cutoff / nyq, btype=filter_type)
    if x.shape[0] < 3 * max(len(a), len(b)):
        return pd.Series(np.nan, x.index, name=x.name)
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
            mat_[:, n] = x**n
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
    if len(x_data) == 0:
        return np.nan
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
    return interpolateNANs(data, "time", gap_limit=inter_limit)


def polynomialInterpolation(data, inter_limit=2, inter_order=2):
    return interpolateNANs(data, "polynomial", gap_limit=inter_limit, order=inter_order)
