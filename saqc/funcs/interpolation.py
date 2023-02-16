#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import UNFLAGGED
from saqc.core import register
from saqc.lib.tools import isflagged
from saqc.lib.ts_operators import interpolateNANs

if TYPE_CHECKING:
    from saqc import SaQC


_SUPPORTED_METHODS = Literal[
    "linear",
    "time",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
]


def _resampleOverlapping(data: pd.Series, freq: str, fill_value):
    """TODO: docstring needed"""
    dtype = data.dtype
    end = data.index[-1].ceil(freq)
    data = data.resample(freq).max()
    data = data.combine(data.shift(1, fill_value=fill_value), max)
    if end not in data:
        data.loc[end] = fill_value
    return data.fillna(fill_value).astype(dtype)


class InterpolationMixin:
    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateByRolling(
        self: "SaQC",
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        """
        Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        window : int, str
            The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
            an string is interpreted as an offset. ( see `pandas.rolling` for more information).
            Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.

        func : Callable
            The function used for aggregation.

        center : bool, default True
            Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.

        min_periods : int
            Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
            computed.

        flag : float or None, default UNFLAGGED
            Flag that is to be inserted for the interpolated values.
            If `None` the old flags are kept, even if the data is valid now.

        Returns
        -------
        saqc.SaQC
        """
        datcol = self._data[field]
        roller = datcol.rolling(window=window, center=center, min_periods=min_periods)
        try:
            func_name = func.__name__
            if func_name[:3] == "nan":
                func_name = func_name[3:]
            rolled = getattr(roller, func_name)()
        except AttributeError:
            rolled = roller.apply(func)

        na_mask = datcol.isna()
        interpolated = na_mask & rolled.notna()
        datcol[na_mask] = rolled[na_mask]
        self._data[field] = datcol

        flagcol = pd.Series(np.nan, index=self._flags[field].index)
        flagcol.loc[interpolated] = np.nan if flag is None else flag

        # todo kwargs must have all passed args except data,field,flags
        meta = {
            "func": "interpolateByRolling",
            "args": (field,),
            "kwargs": {
                "window": window,
                "func": func,
                "center": center,
                "min_periods": min_periods,
                "flag": flag,
                **kwargs,
            },
        }
        self._flags.history[field].append(flagcol, meta)

        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateInvalid(
        self: "SaQC",
        field: str,
        method: _SUPPORTED_METHODS,
        order: int = 2,
        limit: int | None = None,
        extrapolate: Literal["forward", "backward", "both"] = None,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        """
        Function to interpolate nan values in data.

        There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
        the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
            "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
            The interpolation method to use.

        order : int, default 2
            If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
            order.

        limit : int or str, default None
            Upper limit of missing index values (with respect to `freq`) to fill. The limit can either be expressed
            as the number of consecutive missing values (integer) or temporal extension of the gaps to be filled
            (Offset String).
            If `None` is passed, no Limit is set.

        flag : float or None, default UNFLAGGED
            Flag that is set for interpolated values. If ``None``, no flags are set at all.

        downgrade : bool, default False
            If `True` and the interpolation can not be performed at current order, retry with a lower order.
            This can happen, because the chosen ``method`` does not support the passed ``order``, or
            simply because not enough values are present in a interval.

        Returns
        -------
        saqc.SaQC

        Examples
        --------
        See some examples of the keyword interplay below:

        Lets generate some dummy data:

        .. doctest:: interpolateInvalid

           >>> data = pd.DataFrame({'data':np.array([np.nan, 0, np.nan, np.nan, np.nan, 4, 5, np.nan, np.nan, 8, 9, np.nan, np.nan])}, index=pd.date_range('2000',freq='1H', periods=13))
           >>> data
                                data
           2000-01-01 00:00:00   NaN
           2000-01-01 01:00:00   0.0
           2000-01-01 02:00:00   NaN
           2000-01-01 03:00:00   NaN
           2000-01-01 04:00:00   NaN
           2000-01-01 05:00:00   4.0
           2000-01-01 06:00:00   5.0
           2000-01-01 07:00:00   NaN
           2000-01-01 08:00:00   NaN
           2000-01-01 09:00:00   8.0
           2000-01-01 10:00:00   9.0
           2000-01-01 11:00:00   NaN
           2000-01-01 12:00:00   NaN

        Use :py:meth:`~saqc.SaQC.interpolateInvalid` to do linear interpolation of up to 2 consecutive missing values:

        .. doctest:: interpolateInvalid

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.interpolateInvalid("data", limit=3, method='time')
           >>> qc.data # doctest:+NORMALIZE_WHITESPACE
                               data |
           ======================== |
           2000-01-01 00:00:00  NaN |
           2000-01-01 01:00:00  0.0 |
           2000-01-01 02:00:00  NaN |
           2000-01-01 03:00:00  NaN |
           2000-01-01 04:00:00  NaN |
           2000-01-01 05:00:00  4.0 |
           2000-01-01 06:00:00  5.0 |
           2000-01-01 07:00:00  6.0 |
           2000-01-01 08:00:00  7.0 |
           2000-01-01 09:00:00  8.0 |
           2000-01-01 10:00:00  9.0 |
           2000-01-01 11:00:00  NaN |
           2000-01-01 12:00:00  NaN |
           <BLANKLINE>


        Use :py:meth:`~saqc.SaQC.interpolateInvalid` to do linear extrapolaiton of up to 1 consecutive missing values:

        .. doctest:: interpolateInvalid

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.interpolateInvalid("data", limit=2, method='time', extrapolate='both')
           >>> qc.data # doctest:+NORMALIZE_WHITESPACE
                               data |
           ======================== |
           2000-01-01 00:00:00  0.0 |
           2000-01-01 01:00:00  0.0 |
           2000-01-01 02:00:00  NaN |
           2000-01-01 03:00:00  NaN |
           2000-01-01 04:00:00  NaN |
           2000-01-01 05:00:00  4.0 |
           2000-01-01 06:00:00  5.0 |
           2000-01-01 07:00:00  NaN |
           2000-01-01 08:00:00  NaN |
           2000-01-01 09:00:00  8.0 |
           2000-01-01 10:00:00  9.0 |
           2000-01-01 11:00:00  NaN |
           2000-01-01 12:00:00  NaN |
           <BLANKLINE>
        """
        inter_data = interpolateNANs(
            self._data[field],
            method,
            order=order,
            gap_limit=limit,
            extrapolate=extrapolate,
        )

        interpolated = self._data[field].isna() & inter_data.notna()
        self._data[field] = inter_data
        new_col = pd.Series(np.nan, index=self._flags[field].index)
        new_col.loc[interpolated] = np.nan if flag is None else flag

        # todo kwargs must have all passed args except data,field,flags
        self._flags.history[field].append(
            new_col, {"func": "interpolateInvalid", "args": (), "kwargs": kwargs}
        )

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def interpolateIndex(
        self: "SaQC",
        field: str,
        freq: str,
        method: _SUPPORTED_METHODS,
        order: int = 2,
        limit: int | None = 2,
        extrapolate: Literal["forward", "backward", "both"] = None,
        **kwargs,
    ) -> "SaQC":
        """
        Function to interpolate the data at regular (äquidistant) timestamps (or Grid points).

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        freq : str
            An Offset String, interpreted as the frequency of
            the grid you want to interpolate your data at.

        method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
            "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
            The interpolation method you want to apply.

        order : int, default 2
            If your selected interpolation method can be performed at different 'orders' - here you pass the desired
            order.

        limit : int, optional
            Upper limit of missing index values (with respect to `freq`) to fill. The limit can either be expressed
            as the number of consecutive missing values (integer) or temporal extension of the gaps to be filled
            (Offset String).
            If `None` is passed, no Limit is set.

        extraplate : {'forward', 'backward', 'both'}, default None
            Use parameter to perform extrapolation instead of interpolation onto the trailing and/or leading chunks of
            NaN values in data series.

            * 'None' (default) - perform interpolation
            * 'forward'/'backward' - perform forward/backward extrapolation
            * 'both' - perform forward and backward extrapolation

        Returns
        -------
        saqc.SaQC
        """
        if self._data[field].empty:
            return self

        datcol = self._data[field].copy()

        start, end = datcol.index[0].floor(freq), datcol.index[-1].ceil(freq)
        grid_index = pd.date_range(
            start=start, end=end, freq=freq, name=datcol.index.name
        )

        # TODO:
        # in future we could use `register(mask=[field], [], [])`
        # and dont handle masking manually here
        flagged = isflagged(self._flags[field], kwargs["dfilter"])

        # drop all points that hold no relevant grid information
        datcol = datcol[~flagged].dropna()

        # account for annoying case of subsequent frequency aligned values,
        # that differ exactly by the margin of 2*freq
        gaps = datcol.index[1:] - datcol.index[:-1] == 2 * pd.Timedelta(freq)
        gaps = datcol.index[1:][gaps]
        gaps = gaps.intersection(grid_index).shift(-1, freq)

        # prepare grid interpolation:
        datcol = datcol.reindex(datcol.index.union(grid_index))

        # do the grid interpolation
        inter_data = interpolateNANs(
            data=datcol,
            method=method,
            order=order,
            gap_limit=limit,
            extrapolate=extrapolate,
        )

        # override falsely interpolated values:
        inter_data[gaps] = np.nan

        # store interpolated grid
        self._data[field] = inter_data[grid_index]

        history = self._flags.history[field].apply(
            index=self._data[field].index,
            func=_resampleOverlapping,
            func_kws=dict(freq=freq, fill_value=np.nan),
        )

        meta = {
            "func": "interpolateIndex",
            "args": (field,),
            "kwargs": {
                "freq": freq,
                "method": method,
                "order": order,
                "limit": limit,
                "extrapolate": extrapolate,
                **kwargs,
            },
        }
        flagcol = pd.Series(UNFLAGGED, index=history.index)
        history.append(flagcol, meta)

        self._flags.history[field] = history

        return self
