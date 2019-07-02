#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
from typing import Union

import numpy as np
import pandas as pd
import numba as nb

from ..lib.types import PandasLike, ArrayLike


@nb.jit(nopython=True, cache=True)
def findIndex(iterable, value, start):
    i = start
    while i < len(iterable):
        v = iterable[i]
        if v >= value:
            # if v == value:
                # include the end_date if present
                # return i + 1
            return i
        i = i + 1
    return -1


@nb.jit(nopython=True, cache=True)
def valueRange(iterable):
    minval = iterable[0]
    maxval = minval
    for v in iterable:
        if v < minval:
            minval = v
        elif v > maxval:
            maxval = v
    return maxval - minval


def slidingWindowIndices(dates, window_size, iter_delta=None):

    # lets work on numpy data structures for performance reasons
    if isinstance(dates, pd.DataFrame):
        dates = dates.index
    dates = np.array(dates, dtype=np.int64)
    if np.any(np.diff(dates) <= 0):
        raise ValueError("strictly monotic index needed")

    window_size = pd.to_timedelta(window_size, box=False).astype(np.int64)
    if iter_delta:
        iter_delta = pd.to_timedelta(iter_delta, box=False).astype(np.int64)

    start_date = dates[0]
    last_date = dates[-1]
    start_idx = 0
    end_idx = start_idx

    while True:
        end_date = start_date + window_size
        if (end_date > last_date) or (start_idx == -1) or (end_idx == -1):
            break

        end_idx = findIndex(dates, end_date, end_idx)
        yield start_idx, end_idx

        if iter_delta:
            start_idx = findIndex(dates, start_date+iter_delta, start_idx)
        else:
            start_idx += 1

        start_date = dates[start_idx]


def numpyfy(arg:  Union[PandasLike,
                        np.ndarray,
                        numbers.Number]) -> np.ndarray:
    try:
        # pandas dataframe
        return arg.values
    except AttributeError:
        try:
            # numpy array
            return arg.copy()
        except AttributeError:
            # scalar
            return np.atleast_1d(arg)


def broadcastMany(*args: ArrayLike) -> np.ndarray:
    arrays = [np.atleast_1d(a) for a in args]
    target_ndim = max(arr.ndim for arr in arrays)
    out = []
    for arr in arrays:
        out.append(arr[(slice(None),) + (None,) * (target_ndim - arr.ndim)])
    target_shape = np.broadcast(*out).shape
    return tuple(np.broadcast_to(arr, target_shape) for arr in out)


def inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def estimateSamplingRate(index):
    """The function estimates the sampling rate of a datetime index.
    The estimation basically evaluates a histogram of bins with seconds-accuracy. This means, that the
    result may be contra intuitive very likely, if the input series is not rastered (harmonized with skips)
    to an interval divisible by seconds.

    :param index: A DatetimeIndex or array like Datetime listing, of wich you want the sampling rate to be
                  estimated.
    """
    scnds_series = (pd.Series(index).diff().dt.total_seconds()).dropna()
    max_scnds = scnds_series.max()
    min_scnds = scnds_series.min()
    hist = np.histogram(scnds_series, range=(min_scnds, max_scnds + 1), bins=int(max_scnds - min_scnds + 1))
    # return smallest non zero sample difference (this works, because input is expected to be at least
    # harmonized with skips)
    return pd.tseries.frequencies.to_offset(str(int(hist[1][:-1][hist[0] > 0].min())) + 's')
