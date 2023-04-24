#!/usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd


def removeRollingRamps(
    data: pd.Series,
    window: int | str | pd.Timedelta,
    center: bool = False,
    inplace=False,
) -> pd.Series:
    """
    Set data to `NaN`, where the window of prior rolling function was
    not fully shifted into the data.

    Pandas rolling implementation shift a window over the data from left to right.
    The window starts on the far left, outside the data [d], and the window result
    is written [D] on the right edge of the window.

        data:          [d d d d d d ... d d d]
        window(0): [- - D] ==>
        window(1):   [- d D] ==>
        window(2):     [d d D] ==>
           ...                ...
        window(n-1):              ==> [d d D]
        window(n):                  ==> [d d D]

    This results in some windows at the start, where the window is not shifted in
    completely (first two windows in the example above). This might be unexpected or
    undesired for some calculations/algorithms that assume the window is 'filled
    with data'.
    For example: A constant checker that wants to check if a whole day of data is
    within a specific small range, will define a window of `1day`. The data is
    sampled on a 5-minutes base. So the fist window will have one value, the
    second two values, the third three vales etc. until the window is completely
    moved into the data. This might result in the checker marking the beginning of
    the data as constant because it only compares few values. Setting
    ``min_periods`` might help, but for time-based windows which roll over a
    non-regular grid (the time interval between the datapoints is not regular) it
    is not possible to define a value like a 'a full window', because one can not
    know how many (irregular spaced) data-points will be in the window.
    Therefore, this function sets windows, that where not fully shifted into the data
    to ``np.nan``.
    Pass the same arguments for ``window``, ``closed`` and ``center`` as were passed to
    rolling.

    Parameters
    ----------
    data :
        Data resulted from a rolling operation.
    window :
        Pass the same value as was passed to the prior called pd.rolling.
    center :
        Pass the same value as was passed to the prior called pd.rolling.
    closed :
        Pass the same value as was passed to the prior called pd.rolling.
    inplace :
        Return a modified copy if False, otherwise return the modified
        passed in data.

    Returns
    -------
    data: pd.Series
        updated original data or copy

    Examples
    --------

    >>> s = pd.Series(1, index=pd.date_range('2000-02-02', freq='1d', periods=4), dtype=int)
    >>> s
    Out[6]:
    2000-01-01    1
    2000-01-02    1
    2000-01-03    1
    2000-01-04    1
    Freq: D, dtype: int64

    If we roll with a window of 2 days, the result should have NaNs where the window
    is not fully shifted in the data. For the timestamp `2000-02-03 00:00:00` the start
    of the window would be `2000-02-01 00:00:00` which lies outside the data and
    therefore the result is set to NaN.

    >>> dat = s.rolling('2d').sum()
    >>> res = removeRollingRamps(dat, '2d')
    >>> pd.DataFrame(dict(dat=dat, res=res))
                dat  res
    2000-02-02  1.0  NaN
    2000-02-03  2.0  NaN
    2000-02-04  2.0  2.0
    2000-02-05  2.0  2.0

    >>> dat = s.rolling('2d10min').sum()
    >>> res = removeRollingRamps(dat, '2d10min')
    >>> pd.DataFrame(dict(dat=dat, res=res))
                dat  res
    2000-02-02  1.0  NaN
    2000-02-03  2.0  NaN
    2000-02-04  3.0  NaN
    2000-02-05  3.0  3.0

    together with the center keyword the NaNs will appear on both sides of the data
    series, because the window is shifted into the data and also shifts out of it.

    >>> dat = s.rolling('2d', center=True).sum()
    >>> res = removeRollingRamps(dat, '2d', center=True)
    >>> pd.DataFrame(dict(dat=dat, res=res))
                dat  res
    2000-02-02  2.0  NaN
    2000-02-03  2.0  2.0
    2000-02-04  2.0  2.0
    2000-02-05  1.0  NaN
    """
    if not inplace:
        data = data.copy()

    # this ensures index[0] and index[-1]
    if data.empty:
        return data

    if isinstance(window, int):
        window = max(window - 1, 0)
        if center:
            left = math.ceil(window / 2)
            right = math.floor(window / 2)
            data.iloc[:left] = np.nan
            data.iloc[-right:] = np.nan
        else:
            data.iloc[:window] = np.nan
    else:
        window = pd.Timedelta(window)
        if center:
            window //= 2
            data.loc[data.index < data.index[0] + window] = np.nan
            data.loc[data.index > data.index[-1] - window] = np.nan
        else:
            data.loc[data.index < data.index[0] + window] = np.nan
    return data
