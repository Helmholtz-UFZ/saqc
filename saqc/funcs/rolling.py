#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd

from saqc.core import DictOfSeries, Flags, register
from saqc.lib.tools import getFreqDelta

if TYPE_CHECKING:
    from saqc import SaQC


class RollingMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def rolling(
        self: "SaQC",
        field: str,
        window: str | int,
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate a rolling-window function on the data.

        Note, that the data gets assigned the worst flag present in the original data.

        Parameters
        ----------
        window :
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.

        func : default mean
            Function to roll with.

        min_periods :
            The minimum number of periods to get a valid value

        center :
            If True, center the rolling window.
        """
        self._data, self._flags = _roll(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            func=func,
            min_periods=min_periods,
            center=center,
            **kwargs,
        )
        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def roll(
        self: "SaQC",
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate a rolling-window function on the data.

        Note, that the data gets assigned the worst flag present in the original data.

        Parameters
        ----------
        window :
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.

        func : default mean
            Function to roll with.

        min_periods :
            The minimum number of periods to get a valid value

        center :
            If True, center the rolling window.
        """
        import warnings

        warnings.warn(
            """The function `roll` was renamed to `rolling` and will be removed with version 3.0 of saqc
            Please use `SaQC.rolling` with the same arguments, instead
            """,
            DeprecationWarning,
        )
        self._data, self._flags = _roll(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            func=func,
            min_periods=min_periods,
            center=center,
            **kwargs,
        )
        return self


def _roll(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    func: Callable[[pd.Series], np.ndarray] = np.mean,
    min_periods: int = 0,
    center: bool = True,
    **kwargs,
):
    to_fit = data[field].copy()
    if to_fit.empty:
        return data, flags

    regular = getFreqDelta(to_fit.index)
    # starting with the annoying case: finding the rolling interval
    # centers of not-harmonized input time series:
    if center and not regular:
        if isinstance(window, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                'sample series when rolling with "center=True".'
            )
        # get interval centers
        centers = np.floor(
            (
                to_fit.rolling(
                    pd.Timedelta(window) / 2, closed="both", min_periods=min_periods
                ).count()
            )
        )
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        roller = to_fit.rolling(
            pd.Timedelta(window), closed="both", min_periods=min_periods
        )
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(
                pd.Timedelta(window), closed="both", min_periods=min_periods
            ).apply(func)

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = (
            centers.rolling(window, closed="both")
            .apply(center_func, raw=False)
            .astype(int)
        )
        temp = means.copy()
        for k in centers_iloc.iteritems():
            means.iloc[k[1]] = temp[k[0]]
        # last values are false, due to structural reasons:
        means[means.index[centers_iloc[-1]] : means.index[-1]] = np.nan

    # everything is more easy if data[field] is harmonized:
    else:
        if isinstance(window, str):
            window = pd.Timedelta(window) // regular
        if (window % 2 == 0) & center:
            window = int(window - 1)

        roller = to_fit.rolling(window=window, center=center, closed="both")
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(window=window, center=center, closed="both").apply(
                func
            )

    data[field] = means
    worst = flags[field].rolling(window, center=True, min_periods=min_periods).max()
    flags[field] = worst

    return data, flags
