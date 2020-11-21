#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2020, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

# We need to implement the
# - calculation/skipping of min_periods,
# because `calculate_center_offset` does ignore those and we cannot rely on rolling(min_periods), as
# pointed out in customRoller. Also we need to implement
# - centering of windows for fixed windows,
# for variable windows this is not allowed (similar to pandas).
# The close-param, for variable windows is already implemented in `calculate_center_offset`,
# and we dont allow it for fixed windows (similar to pandas). We also want to
# - fix the strange ramp-up behavior,
# which occur if the window is shifted in the data but yet is not fully inside the data. In this
# case we want to spit out nan's instead of results calculated by less than window-size many values.
# This is slightly different than the min_periods parameter, because this mainly should control Nan-behavior
# for fixed windows, and minimum needed observations (also excluding Nans) in a offset window, but should not apply
# if window-size many values couldn't be even possible due to technical reasons. This is mainly because one
# cannot know (except one knows the exact (and fixed) frequency) the number(!) of observations that can occur in a
# given offset window. That's why rolling should spit out Nan's as long as the window is not fully shifted in the data.

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from pandas.core.window.indexers import calculate_variable_window_bounds
from pandas.core.window.rolling import Rolling, calculate_center_offset
from pandas.api.types import is_integer


class _CustomBaseIndexer(BaseIndexer):
    variable_window = None

    def __init__(self, index_array, window_size, min_periods=None, center=False, closed=None, forward=False,
                 expanding=False, step=None, mask=None):
        super().__init__()
        self.index_array = index_array
        self.window_size = window_size
        self.min_periods = min_periods
        self.center = center
        self.closed = closed
        self.forward = forward
        self.expanding = expanding
        self.step = step
        self.skip = mask
        self.num_values = len(self.index_array)
        self.validate()

    def validate(self):
        if not (is_integer(self.window_size) and self.window_size > 0):
            raise TypeError('window_size must be positive integer')

        if self.min_periods is None:
            self.min_periods = 1 if self.variable_window else self.window_size
        if not (is_integer(self.min_periods) and self.min_periods >= 0):
            raise TypeError('min_periods must be non negative integer.')

        if self.variable_window and self.center:
            raise ValueError('offset-based windows cannot be centered.')

        if not self.variable_window and self.closed not in [None, 'right']:
            raise ValueError("For fixed windows only closed='right' is implemented")

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

        assert self.num_values > 0

    def get_window_bounds(self, num_values=0, min_periods=None, center=False, closed=None):
        # do not use the params use ours instead
        num_values = self.num_values
        min_periods = self.min_periods
        center = self.center
        closed = self.closed

        offset = calculate_center_offset(self.window_size) if center else 0
        num_values += offset

        if self.forward:
            start, end = self._fw(num_values, min_periods, center, closed)
        else:
            start, end = self._bw(num_values, min_periods, center, closed)

        if center:
            start, end = self._center_result(start, end, offset)
            num_values -= offset

        if not self.expanding:
            start, end = self._remove_ramps(start, end, offset, center, min_periods)

        if self.skip is not None:
            start, end = self._apply_skipmask(start, end)

        if self.step is not None:
            start, end = self._apply_steps(start, end, num_values)

        start, end = self._mask_min_periods(start, end, num_values)

        return start, end

    def _mask_min_periods(self, start, end, num_values):
        end[end > num_values] = num_values
        end[end - start < self.min_periods] = 0
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
    # set here
    variable_window = False

    def _remove_ramps(self, start, end, offset, center, min_periods):
        fw, bw = self.forward, not self.forward
        rampsz = self.window_size - 1
        if center:
            ramp_l = (rampsz + 1) // 2
            ramp_r = rampsz // 2
            if fw:
                ramp_l, ramp_r = ramp_r, ramp_l
            fw = bw = True
        else:
            ramp_l = ramp_r = rampsz

        if bw and ramp_l > 0:
            end[:ramp_l] = 0
        if fw and ramp_r > 0:
            end[-ramp_r:] = 0

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
    # set here
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
    -----
    If for some reason the start and end numeric indices of the window are needed, one can call
    `start, end = customRoller(obj, ...).window.get_window_bounds()`, which return two arrays,
    holding the start and end indices. Any passed (allowed) parameter to `get_window_bounds()` is
    ignored and instead the arguments that was passed to `customRoller()` beforehand will be evaluated.
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

    kwargs = dict(min_periods=min_periods, center=center, closed=closed, forward=forward, expanding=expanding,
                  step=step, mask=mask, )
    if x.is_freq_type:
        window_indexer = VariableWindowDirectionIndexer(x._on.asi8, x.window, **kwargs)
    else:
        window_indexer = FixedWindowDirectionIndexer(x._on.asi8, window, **kwargs)

    # center offset is calculated from min_periods if a indexer is passed to rolling().
    # if instead a normal window is passed, it is used for offset calculation.
    # also if we pass min_periods == None or 0, all values will exist in the result even if
    # start[i]==end[i]. So we cannot pass the given min_periods to rolling. Instead we set
    # it fix to `1`.
    # this means we need to take care if center=True is passed.
    #
    return obj.rolling(window_indexer, min_periods=1, on=on, axis=axis)
