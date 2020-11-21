#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2020, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from pandas.core.window.indexers import calculate_variable_window_bounds
from pandas.core.window.rolling import Rolling, calculate_center_offset
from pandas.api.types import is_integer


class _CustomBaseIndexer(BaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    forward: bool
    expanding: bool
    skip: np.array
    step: int
    min_periods: int
    variable_window = None

    def __init__(self, index_array, window_size, min_periods=None,
                 forward=False, expanding=False, step=None, mask=None):
        super().__init__(index_array, window_size, min_periods=min_periods,
                         forward=forward, expanding=expanding, step=step, skip=mask)
        self.validate()

    def validate(self):
        if not (is_integer(self.window_size) and self.window_size > 0):
            raise TypeError('window_size must be positive integer')

        if self.min_periods is None:
            self.min_periods = 1 if self.variable_window else self.window_size
        if not (is_integer(self.min_periods) and self.min_periods >= 0):
            raise TypeError('min_periods must be non negative integer.')

        if self.step is not None and not is_integer(self.step):
            raise TypeError('step must be integer.')
        if self.step == 0:
            self.step = None

        if self.skip is not None:
            if len(self.index_array) != len(self.skip):
                raise ValueError('mask must have same length as data.')
            skip = np.array(self.skip)
            if skip.dtype != bool:
                raise TypeError('mask must have boolean values only.')
            self.skip = ~skip

    def _asserts(self, num_values, min_periods, center, closed):
        if self.variable_window:
            assert center is False
        else:
            assert closed in [None, 'right']

    def get_window_bounds(self, num_values=0, min_periods=None, center=False, closed=None):

        self._asserts(num_values, min_periods, center, closed)
        min_periods = self.min_periods

        new_length = num_values
        offset = calculate_center_offset(self.window_size)
        new_length = num_values + offset if center else num_values
        print(locals())

        if self.forward:
            start, end = self._fw(new_length, min_periods, center, closed)
        else:
            start, end = self._bw(new_length, min_periods, center, closed)

        if center:
            start, end = self._center_result(start, end, offset)

        if not self.expanding:
            start, end = self._remove_ramps(start, end, num_values, center, min_periods)

        if self.skip is not None:
            start, end = self._apply_skipmask(start, end)

        if self.step is not None:
            start, end = self._apply_steps(start, end, num_values)

        start, end = self._mask_min_periods(start, end, num_values)

        return start, end

    def _mask_min_periods(self, start, end, num_values):
        end[end > num_values] = num_values
        m = end - start < self.min_periods
        end[m] = 0
        return start, end

    def _center_result(self, start, end, offset):
        if offset > 0:
            start = start[offset:]
            end = end[offset:]
        return start, end

    def _apply_skipmask(self, start, end):
        end[self.skip] = 0
        return start, end

    def _apply_steps(self, start, end, num_values):
        m = np.full(num_values, 0)
        m[::self.step] = 1
        m = m.astype(bool)
        end[m] = 0
        return start, end

    def _remove_ramps(self, start, end, num_values, center, min_periods):
        raise NotImplementedError

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError


class FixedWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    variable_window = False

    def _remove_ramps(self, start, end, num_values, center, min_periods):
        fw, bw = self.forward, not self.forward
        offset_l = offset_r = calculate_center_offset(num_values)
        if center:
            fw = bw = True
            offset_l = (offset_l + 1) // 2
            offset_r = (offset_r + 0) // 2

        offset_l = min(min_periods, offset_l)
        offset_r = min(min_periods, offset_r)
        if bw:
            end[-offset_l:] = 0
        if fw:
            end[:offset_r] = 0

        return start, end

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        # code taken from pd.core.windows.indexer.FixedWindowIndexer
        start_s = np.zeros(self.window_size, dtype="int64")
        start_e = (np.arange(self.window_size, num_values, dtype="int64") - self.window_size + 1)
        start = np.concatenate([start_s, start_e])[:num_values]

        end_s = np.arange(self.window_size, dtype="int64") + 1
        end_e = start_e + self.window_size
        end = np.concatenate([end_s, end_e])[:num_values]
        # end stolen code
        return start, end

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        # code taken from pd.core.windows.indexer.FixedForwardWindowIndexer
        start = np.arange(num_values, dtype="int64")
        end_s = start[: -self.window_size] + self.window_size
        end_e = np.full(self.window_size, num_values, dtype="int64")
        end = np.concatenate([end_s, end_e])
        # end stolen code
        return start, end


class VariableWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    variable_window = True

    def _remove_ramps(self, start, end, num_values, center, min_periods):
        if self.forward:
            # remove (up) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3,  2,  1  ]
            # instead we want:   [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3, nan, nan]
            tresh = self.index_array[-1] - self.window_size
            mask = self.index_array > tresh
        else:
            # remove (down) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min').sum() -> [1,   2,   3, 3, 3]
            # instead we want:   [1,1,1,1,1].rolling(window='2min').sum() -> [nan, nan, 3, 3, 3]
            tresh = self.index_array[0] + self.window_size
            mask = self.index_array < tresh

        end[mask] = 0
        return start, end

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        arr = self.index_array
        start, end = calculate_variable_window_bounds(num_values, self.window_size, min_periods, center, closed, arr)
        return start, end

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        arr = self.index_array[::-1]
        s, _ = calculate_variable_window_bounds(num_values, self.window_size, min_periods, center, closed, arr)
        start = np.arange(num_values)
        end = num_values - s[::-1]
        return start, end


def customRoller(obj, window, min_periods=None,  # aka minimum non-nan values
                 center=False, forward=False, win_type=None, on=None, axis=0, closed=None,
                 expanding=None, step=None, mask=None) -> Rolling:
    """
    A custom rolling implementation, using pandas as base.

    Parameters
    ----------
    obj : pd.Series (or pd.DataFrame)
        The object to roll over. DataFrame is currently still experimental.

    window : int or offset
        Size of the moving window. This is the number of observations used for calculating the statistic.
        Each window will be a fixed size.
        If its an offset then this will be the time period of each window. Each window will be a variable sized
        based on the observations included in the time-period. This is only valid for datetimelike indexes.

    min_periods : int, default None
        Minimum number of observations in window required to have a value (otherwise result is NA).
        For a window that is specified by an offset, min_periods will default to 1. Otherwise, min_periods
        will default to the size of the window.

    center : bool, default False
        Set the labels at the center of the window.

    win_type : str, default None
        Not implemented. Raise NotImplementedError if not None.

    on : str, optional
        For a DataFrame, a datetime-like column or MultiIndex level on which to calculate the rolling window,
        rather than the DataFrame’s index. Provided integer column is ignored and excluded from result since
        an integer index is not used to calculate the rolling window.
    
    axis : int or str, default 0

    closed : str, default None
        Make the interval closed on the ‘right’, ‘left’, ‘both’ or ‘neither’ endpoints. For offset-based windows,
        it defaults to ‘right’. For fixed windows, defaults to ‘both’. Remaining cases not implemented for fixed
        windows.

    forward : bool, default False
        By default a window is 'looking' backwards (in time). If True the window is looking forward in time.

    expanding : bool or None, default None
        If True the window expands/shrink up to its final window size while shifted in the data or shifted out
        respectively.
        For (normal) backward-windows it only expands at the left border, for forward-windows it shrinks on
        the right border and for centered windows both apply.
        For offset-based windows it defaults to False. For fixed windows, defaults to True.
        Also bear in mind that even if this is True, an many as `min_periods` values are necessary to get a
        valid value.


    step : int or None, default None
        If given only every n'th step a window is calculated.

    mask : boolean array-like
        Only calculate the window if the mask is True, otherwise skip it.

    Returns
    -------
    Rolling object: Same as pd.rolling()


    Notes
    """
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        raise TypeError("TODO")
    if win_type is not None:
        raise NotImplementedError("customRoller() not implemented with win_types.")
    try:
        # use .rolling for checks like if center is bool, closed in [left, right, neither, both],
        # center=True is not implemented for offset windows, closed not implemented for integer windows and
        # that the index is monotonic in-/decreasing.
        x = obj.rolling(window=window, min_periods=min_periods, center=center, on=on, axis=axis, closed=closed)
    except Exception:
        raise

    if expanding is None:
        expanding = not x.is_freq_type

    kwargs = dict(forward=forward, expanding=expanding, step=step, mask=mask, min_periods=min_periods)
    if x.is_freq_type:
        window_indexer = VariableWindowDirectionIndexer(x._on.asi8, x.window, **kwargs)
    else:
        window_indexer = FixedWindowDirectionIndexer(x._on.asi8, window, **kwargs)

    # center offset is calculated by min_periods if rolling(indexer)
    # for normal .rolling(window,...) from window instead
    # if min_periods == None or 0, all values exist even if start[i]==end[i]
    # ->> always pass 1
    return obj.rolling(window_indexer, min_periods=1, center=center, on=on, axis=axis, closed=closed)

