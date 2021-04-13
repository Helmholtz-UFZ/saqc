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
from typing import Union
from pandas.api.types import is_integer, is_bool
from pandas.api.indexers import BaseIndexer
from pandas.core.dtypes.generic import ABCSeries, ABCDataFrame
from pandas.core.window.indexers import calculate_variable_window_bounds
from pandas.core.window.rolling import Rolling, Window


def is_slice(k):
    return isinstance(k, slice)


class _CustomBaseIndexer(BaseIndexer):
    is_datetimelike = None

    def __init__(
        self,
        index_array,
        window_size,
        center=False,
        forward=False,
        expand=False,
        step=None,
        mask=None,
    ):
        super().__init__()
        self.index_array = index_array
        self.window_size = window_size
        self._center = center
        self.forward = forward
        self.expand = expand
        self.step = step
        self.skip = mask
        self.validate()

    def validate(self) -> None:
        if self._center is None:
            self._center = False
        if not is_bool(self._center):
            raise ValueError("center must be a boolean")
        if not is_bool(self.forward):
            raise ValueError("forward must be a boolean")
        if not is_bool(self.expand):
            raise ValueError("expand must be a boolean")

        if is_integer(self.step) or self.step is None:
            self.step = slice(None, None, self.step or None)
        if not is_slice(self.step):
            raise TypeError("step must be integer or slice.")
        if self.step == slice(None):
            self.step = None

        if self.skip is not None:
            if len(self.index_array) != len(self.skip):
                raise ValueError("mask must have same length as data to roll over.")
            self.skip = np.array(self.skip)
            if self.skip.dtype != bool:
                raise TypeError("mask must have boolean values only.")
            self.skip = ~self.skip

    def get_window_bounds(
        self, num_values=0, min_periods=None, center=None, closed=None
    ):
        if min_periods is None:
            assert self.is_datetimelike is False
            min_periods = 1

        # if one call us directly, one may pass a center value we should consider.
        # pandas instead (via customRoller) will always pass None and the correct
        # center value is set in __init__. This is because pandas cannot center on
        # dt_like windows and would fail before even call us.
        if center is None:
            center = self._center

        start, end = self._get_bounds(num_values, min_periods, center, closed)

        # ensure correct length
        start, end = start[:num_values], end[:num_values]

        start, end = self._apply_skipmask(start, end)
        start, end = self._apply_steps(start, end, num_values)
        start, end = self._prepare_min_periods_masking(start, end, num_values)
        return start, end

    def _prepare_min_periods_masking(self, start, end, num_values):
        # correction for min_periods calculation
        end[end > num_values] = num_values

        # this is the same as .rolling will do, so leave the work to them ;)
        # additional they are able to count the nans in each window, we couldn't.
        # end[end - start < self.min_periods] = 0
        return start, end

    def _get_center_window_sizes(self, center, winsz):
        ws1 = ws2 = winsz
        if center:
            # centering of dtlike windows is just looking left and right
            # with half amount of window-size
            ws2, ws1 = divmod(winsz, 2)
            ws1 += ws2
            if self.forward:
                ws1, ws2 = ws2, ws1
        return ws1, ws2

    def _apply_skipmask(self, start, end):
        if self.skip is not None:
            end[self.skip] = 0
        return start, end

    def _apply_steps(self, start, end, num_values):
        if self.step is not None:
            m = np.full(num_values, 1)
            m[self.step] = 0
            m = m.astype(bool)
            end[m] = 0
        return start, end

    def _get_bounds(self, num_values=0, min_periods=None, center=False, closed=None):
        raise NotImplementedError


class _FixedWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    # set here
    is_datetimelike = False

    def _get_bounds(self, num_values=0, min_periods=None, center=False, closed=None):
        # closed is always ignored and handled as 'both' other cases not implemented
        offset = 0
        if center:
            offset = (self.window_size - 1) // 2
        num_values += offset

        if self.forward:
            start, end = self._fw(num_values, offset)
        else:
            start, end = self._bw(num_values, offset)

        if center:
            start, end = self._center_result(start, end, offset)
            num_values -= offset

        if not self.expand:
            start, end = self._remove_ramps(start, end, center)

        return start, end

    def _center_result(self, start, end, offset):
        # cut N values at the front that was inserted in _fw()
        # or cut N values at the end if _bw()
        if offset > 0:
            if self.forward:
                start = start[:-offset]
                end = end[:-offset]
            else:
                start = start[offset:]
                end = end[offset:]
        return start, end

    def _remove_ramps(self, start, end, center):
        fw, bw = self.forward, not self.forward
        ramp_l, ramp_r = self._get_center_window_sizes(center, self.window_size - 1)
        if center:
            fw = bw = True

        if bw and ramp_l > 0:
            end[:ramp_l] = 0
        if fw and ramp_r > 0:
            end[-ramp_r:] = 0

        return start, end

    def _bw(self, num_values=0, offset=0):
        start = np.arange(-self.window_size, num_values + offset, dtype="int64") + 1
        end = start + self.window_size
        start[: self.window_size] = 0
        return start, end

    def _fw(self, num_values=0, offset=0):
        start = np.arange(-offset, num_values, dtype="int64")
        end = start + self.window_size
        start[:offset] = 0
        return start, end


class _VariableWindowDirectionIndexer(_CustomBaseIndexer):
    # automatically added in super call to init
    index_array: np.array
    window_size: int
    # set here
    is_datetimelike = True

    def _get_bounds(self, num_values=0, min_periods=None, center=False, closed=None):
        ws_bw, ws_fw = self._get_center_window_sizes(center, self.window_size)
        if center:
            c1 = c2 = closed
            if closed == "neither":
                c1, c2 = "right", "left"

            start, _ = self._bw(num_values, ws_bw, c1)
            _, end = self._fw(num_values, ws_fw, c2)

        elif self.forward:
            start, end = self._fw(num_values, ws_fw, closed)
        else:
            start, end = self._bw(num_values, ws_bw, closed)

        if not self.expand:
            start, end = self._remove_ramps(start, end, center)

        return start, end

    def _remove_ramps(self, start, end, center):
        ws_bw, ws_fw = self._get_center_window_sizes(center, self.window_size)

        if center or not self.forward:
            # remove (up) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min').sum() -> [1,   2,   3, 3, 3]
            # instead we want:   [1,1,1,1,1].rolling(window='2min').sum() -> [nan, nan, 3, 3, 3]
            tresh = self.index_array[0] + ws_bw
            mask = self.index_array < tresh
            end[mask] = 0

        if center or self.forward:
            # remove (down) ramp
            # we dont want this: [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3,  2,  1  ]
            # instead we want:   [1,1,1,1,1].rolling(window='2min', forward=True).sum() -> [3, 3, 3, nan, nan]
            tresh = self.index_array[-1] - ws_fw
            mask = self.index_array > tresh
            end[mask] = 0

        return start, end

    def _bw(self, num_values, window_size, closed):
        arr = self.index_array
        start, end = calculate_variable_window_bounds(
            num_values, window_size, None, None, closed, arr
        )
        return start, end

    def _fw(self, num_values, window_size, closed):
        arr = self.index_array[::-1]
        s, _ = calculate_variable_window_bounds(
            num_values, window_size, None, None, closed, arr
        )
        start = np.arange(num_values)
        end = num_values - s[::-1]

        if closed in ["left", "neither"]:
            start += 1
        return start, end


def customRoller(
    obj,
    window,
    min_periods=None,  # aka minimum non-nan values
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
    forward=False,
    expand=True,
    step=None,
    mask=None,
) -> Union[Rolling, Window]:
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
        Set the labels at the center of the window. Also works for offset-based windows (in contrary to pandas).

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

    expand : bool, default True
        If True the window expands/shrink up to its final window size while shifted in the data or shifted out
        respectively.
        For (normal) backward-windows it only expands at the left border, for forward-windows it shrinks on
        the right border and for centered windows both apply.

        Also bear in mind that even if this is True, an many as `min_periods` values are necessary to get a
        valid value, see there for more info.


    step : int, slice or None, default None
        If given, only every n'th step a window is calculated starting from the very first. One can
        give a slice if one want to start from eg. the second (`slice(2,None,n)`) or similar.

    mask : boolean array-like
        Only calculate the window if the mask is True, otherwise skip it.

    Returns
    -------
    a Window or Rolling sub-classed for the particular operation


    Notes
    -----
    If for some reason the start and end numeric indices of the window are needed, one can call
    `start, end = customRoller(obj, window).window.get_window_bounds(num_values, min_periods)`,
    which return two np.arrays, that are holding the start and end indices. Fill at least all
    parameter which are shown in the example.

    See Also
    --------
    pandas.Series.rolling
    pandas.DataFrame.rolling
    """
    num_params = len(locals()) - 2  # do not count window and obj
    if not isinstance(obj, (ABCSeries, ABCDataFrame)):
        raise TypeError(f"invalid type: {type(obj)}")

    # center is the only param from the pandas rolling implementation
    # that we advance, namely we allow center=True on dt-indexed data
    # that's why we take it as ours
    theirs = dict(
        min_periods=min_periods, win_type=win_type, on=on, axis=axis, closed=closed
    )
    ours = dict(center=center, forward=forward, expand=expand, step=step, mask=mask)
    assert len(theirs) + len(ours) == num_params, "not all params covert (!)"

    # use .rolling to do all the checks like if closed is one of [left, right, neither, both],
    # closed not allowed for integer windows, index is monotonic (in- or decreasing), if freq-based
    # windows can be transformed to nanoseconds (eg. fails for `1y` - it could have 364 or 365 days), etc.
    # Also it converts window and the index to numpy-arrays (so we don't have to do it :D).
    try:
        x = obj.rolling(window, **theirs)
    except Exception:
        raise

    indexer = (
        _VariableWindowDirectionIndexer
        if x.is_freq_type
        else _FixedWindowDirectionIndexer
    )
    indexer = indexer(index_array=x._on.asi8, window_size=x.window, **ours)

    # Centering is fully done in our own indexers. So we do not pass center to rolling(). Especially because
    # we also allow centering on dt-based indexes. Also centering would fail in forward windows, because of
    # pandas internal centering magic (append nans at the end of array, later cut values from beginning of the
    # result).
    # min_periods is also quite tricky. Especially if None is passed. For dt-based windows min_periods defaults to 1
    # and is set during rolling setup (-> if r=obj.rolling() is called). For numeric windows instead, it keeps None
    # during setup and defaults to indexer.window_size if a rolling-method is called (-> if r.sum()). Thats a bit
    # odd and quite hard to find. So we are good if we pass the already calculated x.min_periods as this will just
    # hold the correct initialised or not initialised value. (It gets even trickier if one evaluates which value is
    # actually passed to the function that actually thrown them out; i leave that to the reader to find out. start
    # @ pandas.core.window.rolling:_Window._apply)
    # Lastly, it is necessary to pass min_periods at all (!) and do not set it to a fix value (1, 0, None,...). This
    # is, because we cannot throw out values by ourself in the indexer, because min_periods also evaluates NA values
    # in its count and we have no control over the actual values, just their indexes.
    theirs.update(min_periods=x.min_periods)
    roller = obj.rolling(indexer, center=None, **theirs)

    # ----- count hack -------
    # Unfortunately pandas calls count differently if a BaseIndexer
    # instance is given. IMO, the intention behind this is to call
    # count different for dt-like windows, but if a user pass a own
    # indexer we also end up in this case /:
    # The only possibility is to monkey-patch pandas...
    def new_count():
        self = roller
        if not x.is_freq_type:
            obj_new = obj.notna().astype(int)
            if min_periods is None:
                theirs.update(min_periods=0)
            return obj_new.rolling(indexer, center=None, **theirs).sum()
        return self._old_count()

    roller._old_count = roller.count
    roller.count = new_count
    # ----- count hack -------

    return roller
