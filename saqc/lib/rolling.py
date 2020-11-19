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

    def __init__(self, index_array, window_size, forward=False, expanding=False, step=None, mask=None):
        super().__init__(index_array, window_size, forward=forward, expanding=expanding, step=step, skip=mask)
        self.validate()

    def validate(self):
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise TypeError('window_size must be positive integer')
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

    def get_window_bounds(self, num_values=0, min_periods=None, center=False, closed=None):

        # only for consistency. min_periods is None if and only
        # if we have a fixed window and the user did not passed it.
        # Nevertheless min_periods is not used so far.
        if min_periods is None:
            min_periods = self.window_size

        if self.forward:
            start, end = self._fw(num_values, min_periods, center, closed)
        else:
            start, end = self._bw(num_values, min_periods, center, closed)

        if not self.expanding:
            start, end = self._remove_ramp(start, end)

        if self.skip is not None:
            start, end = self._apply_skipmask(start, end)

        if self.step is not None:
            start, end = self._apply_steps(start, end, num_values)

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

    def _remove_ramp(self, start, end, center=False):
        fw, bw = self.forward, not self.forward
        winsz = self.window_size
        if center:
            winsz //= 2

        if center or fw:
            # remove (up) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3,  2,  1  ]
            # instead we want:   [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3, nan, nan]
            tresh = self.index_array[-1] - winsz
            mask = self.index_array > tresh
            end[mask] = 0

        if center or bw:
            # remove (down) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min').sum() -> [1,   2,   3, 3, 3]
            # instead we want:   [1,1,1,1,1].rolling(window='2min').sum() -> [nan, nan, 3, 3, 3]
            tresh = self.index_array[0] + winsz
            mask = self.index_array < tresh
            end[mask] = 0

        return start, end

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError


class _FixedWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int

    def _raise_if_closed(self, closed):
        if closed is None or closed == 'right':
            return
        raise ValueError(f"closed other than 'right' is not implemented for fixed windows.")

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        self._raise_if_closed(closed)

        offset = calculate_center_offset(self.window_size)
        num_values += offset if center else 0

        # code taken from pd.core.windows.indexer.FixedWindowIndexer
        start_s = np.zeros(self.window_size, dtype="int64")
        start_e = (np.arange(self.window_size, num_values, dtype="int64") - self.window_size + 1)
        start = np.concatenate([start_s, start_e])[:num_values]

        end_s = np.arange(self.window_size, dtype="int64") + 1
        end_e = start_e + self.window_size
        end = np.concatenate([end_s, end_e])[:num_values]
        # end stolen code

        if center and offset > 0:
            start = start[offset:]
            end = end[offset:]
            end[-offset:] = 0

        return start, end

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        self._raise_if_closed(closed)

        offset = calculate_center_offset(self.window_size)
        num_values += offset if center else 0

        # code taken from pd.core.windows.indexer.FixedForwardWindowIndexer
        start = np.arange(num_values, dtype="int64")
        end_s = start[: -self.window_size] + self.window_size
        end_e = np.full(self.window_size, num_values, dtype="int64")
        end = np.concatenate([end_s, end_e])
        # end stolen code

        if center and offset > 0:
            start = start[:offset]
            end = end[:offset]
            end[:offset] = 0

        return start, end


class _VariableWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int

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

    kwargs = dict(forward=forward, expanding=expanding, step=step, mask=mask)
    if x.is_freq_type:
        window_indexer = _VariableWindowDirectionIndexer(x._on.asi8, x.window, **kwargs)
    else:
        window_indexer = _FixedWindowDirectionIndexer(x._on.asi8, window, **kwargs)

    return obj.rolling(window_indexer, min_periods=x.min_periods, center=center, on=on, axis=axis, closed=closed)


if __name__ == '__main__':
    s1 = pd.Series(1., index=pd.date_range("1999/12", periods=12, freq='1M') + pd.Timedelta('1d'))
    s2 = pd.Series(1., index=pd.date_range('2000/05/15', periods=8, freq='1d'))
    s = pd.concat([s1, s2]).sort_index()
    s.name = 's'
    s[15] = np.nan

    df = pd.DataFrame(s)
    df['32d'] = s.rolling('32d').sum()
    df['C-32d'] = customRoller(s, '32d').sum()
    df['32d-mp2'] = s.rolling('32d', min_periods=2).sum()
    df['C-32d-mp2'] = customRoller(s, '32d', min_periods=2).sum()
    print('\n', df)

    df = pd.DataFrame(s)
    df['ffhack-32d'] = pd.Series(reversed(s), reversed(s.index)).rolling('32d').sum()[::-1]
    df['Cff-32d'] = customRoller(s, '32d', forward=True).sum()
    df['ffhack-32d-mp2'] = pd.Series(reversed(s), reversed(s.index)).rolling('32d', min_periods=2).sum()[::-1]
    df['Cff-32d-mp2'] = customRoller(s, '32d', min_periods=2, forward=True).sum()
    print('\n', df)

    df = pd.DataFrame(s)
    df['4'] = s.rolling(4).sum()
    df['C4'] = customRoller(s, 4).sum()
    df['4-mp2'] = s.rolling(4, min_periods=2).sum()
    df['C4-mp2'] = customRoller(s, 4, min_periods=2).sum()
    print('\n', df)

    df = pd.DataFrame(s)
    df['ffhack4'] = pd.Series(reversed(s), reversed(s.index)).rolling(4).sum()[::-1]
    df['Cff-4'] = customRoller(s, 4, forward=True).sum()
    df['ffhack4-mp2'] = pd.Series(reversed(s), reversed(s.index)).rolling(4, min_periods=2).sum()[::-1]
    df['Cff-4-mp2'] = customRoller(s, 4, min_periods=2, forward=True).sum()
    df['ffhack4-mp1'] = pd.Series(reversed(s), reversed(s.index)).rolling(4, min_periods=1).sum()[::-1]
    df['Cff-4-mp1'] = customRoller(s, 4, min_periods=1, forward=True).sum()
    print('\n', df)

    df = pd.DataFrame(s)
    mask = [True, False] * int(len(s) / 2)
    df['m[TF]'] = customRoller(s, 4, mask=mask).sum()
    df['stp=3'] = customRoller(s, 4, step=3).sum()
    df['m[TF]stp=3'] = customRoller(s, 4, step=3, mask=mask).sum()
    df['ff-m[TF]'] = customRoller(s, 4, forward=True, mask=mask).sum()
    df['ff-stp=3'] = customRoller(s, 4, forward=True, step=3).sum()
    df['ff-m[TF]stp=3'] = customRoller(s, 4, forward=True, step=3, mask=mask).sum()
    print('\n', df)

    s = pd.Series(1., index=pd.date_range('2000/05/15', periods=10, freq='1d'))
    df = pd.DataFrame(s)
    df['3d-right'] = s.rolling('3d', closed='right').sum()
    df['C3d-right'] = customRoller(s, '3d', closed='right').sum()
    df['3d-left'] = s.rolling('3d', closed='left').sum()
    df['C3d-left'] = customRoller(s, '3d', closed='left').sum()
    print('\n', df)

    df = pd.DataFrame(s)
    df['3d-both'] = s.rolling('3d', closed='both').sum()
    df['C3d-both'] = customRoller(s, '3d', closed='both').sum()
    df['3d-neither'] = s.rolling('3d', closed='neither').sum()
    df['C3d-neither'] = customRoller(s, '3d', closed='neither').sum()
    print('\n', df)

    s[6] = 0
    df = pd.DataFrame(s)
    df['a'] = s.rolling(7, center=False).sum()
    df['a`'] = customRoller(s, 7, center=False).sum()

    df = pd.DataFrame(s)
    df['Cffc'] = customRoller(s, 8, forward=True, center=True).sum()
    print('\n', df)
    # exit(3)

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
        df = pd.DataFrame(s)
        print(i, (i - 1) // 2)
        df['c'] = s.rolling(i, center=True).sum()
        df['Cc'] = customRoller(s, i, center=True).sum()
        df['Cffc'] = customRoller(s, i, forward=True, center=True).sum()
        print('\n', df)

