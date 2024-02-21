#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from saqc import BAD
from saqc.core import flagging
from saqc.lib.checking import validateMinPeriods, validateValueBounds, validateWindow
from saqc.lib.rolling import removeRollingRamps
from saqc.lib.tools import getFreqDelta, statPass
from saqc.lib.ts_operators import varQC

if TYPE_CHECKING:
    from saqc import SaQC


class ConstantsMixin:
    @flagging()
    def flagConstants(
        self: "SaQC",
        field: str,
        thresh: float,
        window: int | str,
        min_periods: int = 2,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag constant data values.

        Flags plateaus of constant data if their maximum total change in
        a rolling window does not exceed a certain threshold.

        Any interval of values y(t),...,y(t+n) is flagged, if:
         - (1): n > ``window``
         - (2): abs(y(t + i) - (t + j)) < `thresh`, for all i,j in [0, 1, ..., n]

        Parameters
        ----------
        thresh :
            Maximum total change allowed per window.

        window :
            Size of the moving window. This determines the number of observations used
            for calculating the absolute change per window.
            Each window will either contain a fixed number of periods (integer defined window),
            or will have a fixed temporal extension (offset defined window).

        min_periods :
            Minimum number of observations in window required to generate
            a flag. This can be used to exclude underpopulated *offset* defined windows from
            flagging. (Integer defined windows will always contain exactly *window* samples).
            Must be an integer greater or equal `2`, because a
            single value would always be considered constant.
            Defaults to `2`.
        """
        d: pd.Series = self._data[field]
        validateWindow(window, index=d.index)
        validateMinPeriods(min_periods, minimum=2, optional=False)

        # 1. find starting points of consecutive constant values as a boolean mask
        # 2. fill the whole window with True's
        rolling = d.rolling(window=window, min_periods=min_periods)
        starting_points_mask = rolling.max() - rolling.min() <= thresh

        starting_points_mask = removeRollingRamps(starting_points_mask, window=window)

        # mimic forward rolling by roll over inverse [::-1]
        rolling = starting_points_mask[::-1].rolling(
            window=window, min_periods=min_periods
        )
        # mimic any()
        mask = (rolling.sum()[::-1] > 0) & d.notna()

        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagByVariance(
        self: "SaQC",
        field: str,
        window: str,
        thresh: float,
        maxna: int | None = None,
        maxna_group: int | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag low-variance data.

        Flags plateaus of constant data if the variance in a rolling window does not
        exceed a certain threshold.

        Any interval of values y(t),..y(t+n) is flagged, if:

        (1) n > `window`
        (2) variance(y(t),...,y(t+n) < `thresh`

        Parameters
        ----------
        window :
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window.
            Each window will be sized, based on the number of observations included
            in the time-period.

        thresh :
            Maximum total variance allowed per window.

        maxna :
            Maximum number of NaNs allowed in window.
            If more NaNs are present, the window is not flagged.

        maxna_group :
            Same as `maxna` but for consecutive NaNs.
        """
        d: pd.Series = self._data[field]
        validateWindow(window, allow_int=False, index=d.index)
        window = pd.Timedelta(window)

        delta = getFreqDelta(d.index)
        if not delta:
            raise IndexError("Timeseries irregularly sampled!")

        if maxna is None:
            maxna = np.inf
        if maxna_group is None:
            maxna_group = np.inf

        validateValueBounds(maxna, "maxna", 0, closed="both", strict_int=True)
        validateValueBounds(
            maxna_group, "maxna_group", 0, closed="both", strict_int=True
        )

        min_periods = int(np.ceil(pd.Timedelta(window) / pd.Timedelta(delta)))
        to_set = statPass(
            d,
            lambda x: varQC(x, maxna, maxna_group),
            window,
            thresh,
            operator.lt,
            min_periods=min_periods,
        )

        self._flags[to_set, field] = flag
        return self
