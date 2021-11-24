#!/usr/bin/env python

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from pandas.api.types import is_bool

if pd.__version__ < "1.4":
    import pandas.core.window.indexers as indexers
else:
    import pandas.core.indexers.objects as indexers


class ForwardMixin:
    def __init__(self, *args, forward: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = forward

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if closed is None:
            closed = "right"

        if self.forward:

            # this is only set with variable window indexer
            if self.index_array is not None:  # noqa
                self.index_array = self.index_array[::-1]  # noqa

            # closed 'both' and 'neither' works the same
            # on forward and backward windows by definition
            if closed == "left":
                closed = "right"
            elif closed == "right":
                closed = "left"

        start, end = super().get_window_bounds(  # noqa
            num_values, min_periods, center, closed
        )

        if self.forward:
            start, end = end, start
            start = num_values - start[::-1]
            end = num_values - end[::-1]

        return start, end


CustomFixedIndexer = type(
    "CustomFixedIndexer", (ForwardMixin, indexers.FixedWindowIndexer), {}
)

CustomVariableIndexer = type(
    "CustomVariableIndexer", (ForwardMixin, indexers.VariableWindowIndexer), {}
)


class AttrWrapper(object):
    """
    This wraps a attribute like `rolling.closed` to `customRoller.closed`.
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return getattr(instance._roller, self.name)

    def __set__(self, instance, value):
        setattr(instance._roller, self.name, value)


class CustomRoller:
    def __init__(
        self,
        obj: pd.DataFrame | pd.Series,
        window: int | str | pd.Timedelta,
        min_periods: int | None = None,  # aka minimum non-nan values
        center: bool | None = False,
        win_type: str | None = None,
        on: str | None = None,
        axis: int | str = 0,
        closed: str | None = None,
        forward: bool = False,
        expand=None,
    ):
        """
        A custom rolling implementation, using pandas as base.

        Parameters
        ----------
        obj : pd.Series (or pd.DataFrame)
            The object to roll over. DataFrame is currently still experimental.

        window : int or offset
            Size of the moving window. This is the number of observations used for
            calculating the statistic. Each window will be a fixed size. If its an
            offset then this will be the time period of each window. Each window will
            be a variable sized based on the observations included in the
            time-period. This is only valid for datetimelike indexes.

        min_periods : int, default None
            Minimum number of observations in window required to have a value (
            otherwise result is NA). For a window that is specified by an offset,
            min_periods will default to 1. Otherwise, min_periods will default to the
            size of the window.

        center : bool, default False
            Set the labels at the center of the window. Also works for offset-based
            windows (in contrary to pandas).

        win_type : str, default None
            Not implemented. Raise NotImplementedError if not None.

        on : str, optional
            For a DataFrame, a datetime-like column or MultiIndex level on which to
            calculate the rolling window, rather than the DataFrame’s index. Provided
            integer column is ignored and excluded from result since an integer index
            is not used to calculate the rolling window.

        axis : int or str, default 0

        closed : str, default None
            Make the interval closed on the `right`, `left`, `both` or `neither`
            endpoints. For offset-based windows, with ``forward=False`` it defaults
            to `right`, for ``forward=True`` it defaults to `left`.
            For fixed windows, defaults to ‘both’ always.

        forward : bool, default False
            By default a window is 'looking' backwards (in time). If True the window
            is looking forward in time.

        expand : bool, default True
            If True the window expands/shrink up to its final window size while
            shifted in the data or shifted out respectively. For (normal)
            backward-windows it only expands at the left border, for forward-windows
            it shrinks on the right border and for centered windows both apply.

            Also bear in mind that even if this is True, an many as `min_periods` values are necessary to get a
            valid value, see there for more info.

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

        if not is_bool(forward):
            raise ValueError("forward must be a boolean")

        # only relevant for datetime-like windows
        if expand is None:
            if min_periods is None:
                expand = False
                warnings.warn(
                    "`expand` defaults to False, if min_periods is None. The result "
                    "will differ from pandas rolling implementation. To fallback to "
                    "pandas rolling, use `expand=True`, to silence this warning and "
                    "use our rolling use `expand=False` or specify `min_periods`."
                )
            else:
                expand = True

        if not is_bool(expand):
            raise TypeError(f"expand must be bool or None not {type(expand).__name__}")

        # ours
        self._forward = forward
        self._expand = expand

        # dummy roller.
        # This lets pandas do all the checks.
        verified = obj.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
        )

        self._dtlike_window = verified._win_freq_i8 is not None

        # these roller attributes are fixed by us and not
        # get looked up from self._roller in __getattr__
        # because they might not be correct when passing
        # a custom indexer to rolling
        self.window = verified.window
        self._win_freq_i8 = verified._win_freq_i8
        self._index_array = verified._index_array

        if self._dtlike_window:
            self.window_indexer = CustomVariableIndexer(
                index_array=verified._index_array,
                window_size=verified._win_freq_i8,
                center=verified.center,
                forward=self._forward,
            )
        else:
            self.window_indexer = CustomFixedIndexer(
                window_size=verified.window,
                forward=self._forward,
            )

        # set the default
        if self._forward and closed is None:
            closed = "left"

        # create the real roller with a custom Indexer
        # from the attributes of the old roller.
        # After next line, all the attributes (public and private)
        #    of `_roller` are accessible on self
        self._roller = obj.rolling(
            window=self.window_indexer,
            min_periods=verified.min_periods,  # roller.min_periods
            win_type=verified._win_type,  # use private here to silence warning
            on=verified.on,
            center=verified.center,
            closed=closed,
            axis=verified.axis,
        )

    def _call_roll_func(self, name, args, kwargs):
        result = getattr(self._roller, name)(*args, **kwargs)
        if self._dtlike_window and not self._expand:
            result = self._remove_expanding_ramps(result)
        return result

    def _remove_expanding_ramps(self, result):
        if len(result.index) == 0:
            return result

        index = self._index_array
        window_size = self._win_freq_i8
        decreasing = index[0] > index[-1]
        mask = np.full_like(index, False, dtype=bool)

        if self.center:
            window_size //= 2

        if self.center or self._forward:
            if decreasing:
                mask |= index < index[-1] + window_size
            else:
                mask |= index > index[-1] - window_size

        if self.center or not self._forward:
            if decreasing:
                mask |= index > index[0] - window_size
            else:
                mask |= index < index[0] + window_size

        if window_size > 0:
            result[mask] = np.nan

        return result

    # =========================================================================
    # public interface
    #
    # All attributes of roller are accessible on self.
    # Public attributes are listed below, for the only reason to provide an
    # autocompletion for them.
    # Private attributes are `wrapped` via ``__getattr__``
    # =========================================================================

    def __getattr__(self, item):
        return getattr(self._roller, item)

    obj = AttrWrapper("obj")
    closed = AttrWrapper("closed")
    center = AttrWrapper("center")
    axis = AttrWrapper("axis")
    exclusions = AttrWrapper("exclusions")
    is_datetimelike = AttrWrapper("is_datetimelike")
    is_freq_type = AttrWrapper("is_freq_type")
    min_periods = AttrWrapper("min_periods")
    ndim = AttrWrapper("ndim")
    on = AttrWrapper("on")
    sparse = AttrWrapper("sparse")
    win_freq = AttrWrapper("win_freq")
    win_type = AttrWrapper("win_type")
    method = AttrWrapper("method")

    def sum(self, *args, **kwargs):
        return self._call_roll_func("sum", args, kwargs)

    def count(self, *args, **kwargs):
        return self._call_roll_func("count", args, kwargs)

    def mean(self, *args, **kwargs):
        return self._call_roll_func("mean", args, kwargs)

    def median(self, *args, **kwargs):
        return self._call_roll_func("median", args, kwargs)

    def min(self, *args, **kwargs):
        return self._call_roll_func("min", args, kwargs)

    def max(self, *args, **kwargs):
        return self._call_roll_func("max", args, kwargs)

    def skew(self, *args, **kwargs):
        return self._call_roll_func("skew", args, kwargs)

    def kurt(self, *args, **kwargs):
        return self._call_roll_func("kurt", args, kwargs)

    def var(self, *args, **kwargs):
        return self._call_roll_func("var", args, kwargs)

    def std(self, *args, **kwargs):
        return self._call_roll_func("std", args, kwargs)

    def sem(self, *args, **kwargs):
        return self._call_roll_func("sem", args, kwargs)

    def quantile(self, *args, **kwargs):
        return self._call_roll_func("quantile", args, kwargs)

    def cov(self, *args, **kwargs):
        return self._call_roll_func("cov", args, kwargs)

    def corr(self, *args, **kwargs):
        return self._call_roll_func("corr", args, kwargs)

    def apply(self, *args, **kwargs):
        return self._call_roll_func("apply", args, kwargs)

    def aggregate(self, *args, **kwargs):
        return self._call_roll_func("aggregate", args, kwargs)

    agg = aggregate

    def validate(self):  # dummy function to fit Rolling class
        self._roller.validate()


customRoller = CustomRoller
