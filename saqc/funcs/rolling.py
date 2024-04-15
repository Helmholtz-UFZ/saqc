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
from saqc.lib.checking import (
    validateCallable,
    validateFuncSelection,
    validateMinPeriods,
    validateWindow,
)
from saqc.lib.tools import getFreqDelta

if TYPE_CHECKING:
    from saqc import SaQC


class RollingMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def rolling(
        self: "SaQC",
        field: str,
        window: str | int,
        func: Callable[[pd.Series], np.ndarray] | str = "mean",
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
        # HINT: checking in  _roll
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
    func: Callable[[pd.Series], np.ndarray] | str = "mean",
    min_periods: int = 0,
    center: bool = True,
    **kwargs,
):
    validateFuncSelection(func, allow_operator_str=True)
    validateWindow(window)
    validateMinPeriods(min_periods)

    to_fit = data[field].copy()
    flags_col = flags[field].copy()
    if to_fit.empty:
        return data, flags

    d_roller = to_fit.rolling(window, min_periods=min_periods, center=center)
    if isinstance(func, str):
        to_fit = getattr(d_roller, func)()
    else:
        to_fit = d_roller.apply(func)

    flags_col = flags_col.rolling(window, min_periods=min_periods, center=center).max()
    data[field] = to_fit
    flags[field] = flags_col
    return data, flags
