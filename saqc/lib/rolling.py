#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2020, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from pandas.core.window.indexers import calculate_variable_window_bounds
from pandas.core.window.rolling import Rolling, VariableWindowIndexer
from typing import *


class _RmRampIndexer(BaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    forward: bool
    rm_ramp: bool

    def __init__(self, index_array, window_size, forward=False, rm_ramp=True):
        super().__init__(index_array, window_size, forward=forward, rm_ramp=rm_ramp)

    def get_window_bounds(self, num_values=0, min_periods=None, center=False, closed=None):
        if self.forward:
            start, end = self._fw(num_values, min_periods, center, closed)
        else:
            start, end = self._bw(num_values, min_periods, center, closed)

        if self.rm_ramp:
            start, end = self._remove_ramp(start, end)

        return start, end

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError

    def _remove_ramp(self, start, end):
        if self.forward:
            # remove (up) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3,  2,  1  ]
            # instead we want:   [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3, nan, nan]
            tresh = self.index_array[-1] - self.window_size
            mask = self.index_array > tresh
            start[mask], end[mask] = 0, 0
        else:
            # remove (down) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min').sum() -> [1,   2,   3, 3, 3]
            # instead we want:   [1,1,1,1,1].rolling(window='2min').sum() -> [nan, nan, 3, 3, 3]
            tresh = self.index_array[0] + self.window_size
            mask = self.index_array < tresh
            start[mask], end[mask] = 0, 0
        return start, end


class _FixedWindowDirectionIndexer(_RmRampIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    forward: bool
    rm_ramp: bool

    def _bw(self, num_values=0, min_periods=None, center=False, closed=None):
        # code taken from pd.core.windows.indexer.FixedWindowIndexer
        start_s = np.zeros(self.window_size, dtype="int64")
        start_e = (np.arange(self.window_size, num_values, dtype="int64") - self.window_size + 1)
        start = np.concatenate([start_s, start_e])[:num_values]

        end_s = np.arange(self.window_size, dtype="int64") + 1
        end_e = start_e + self.window_size
        end = np.concatenate([end_s, end_e])[:num_values]
        return start, end

    def _fw(self, num_values=0, min_periods=None, center=False, closed=None):
        # code taken from pd.core.windows.indexer.FixedForwardWindowIndexer
        start = np.arange(num_values, dtype="int64")
        end_s = start[: -self.window_size] + self.window_size
        end_e = np.full(self.window_size, num_values, dtype="int64")
        end = np.concatenate([end_s, end_e])
        return start, end


class _VariableWindowDirectionIndexer(_RmRampIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    forward: bool
    rm_ramp: bool

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
                 rm_ramp=True) -> Rolling:
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

    kwargs = dict(forward=forward, rm_ramp=rm_ramp)
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
