#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from pydantic import validate_call

from saqc import BAD
from saqc.core import flagging
from saqc.lib.rolling import removeRollingRamps
from saqc.lib.tools import getFreqDelta, statPass
from saqc.lib.ts_operators import varQC
from saqc.lib.types import Float, Int, OffsetStr, SaQC, ValidatePublicMembers


class ConstantsMixin(ValidatePublicMembers):
    @flagging()
    def flagConstants(
        self: SaQC,
        field: str,
        thresh: Float >= 0,
        window: OffsetStr | (Int >= 1),
        min_periods: Int >= 0 = 2,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Flag constant data values.

        Flags plateaus of constant data if their maximum total change in a rolling
        window does not exceed a certain threshold.

        Any interval of values y(t), ..., y(t+n) is flagged if:
         - n > window
         - abs(y(t + i) - y(t + j)) < thresh for all i, j in [0, 1, ..., n]

        Parameters
        ----------
        thresh : float
            Maximum total change allowed per window.

        window : int or str
            Size of the rolling window. If an integer is passed, it represents the number
            of timestamps per window. If an offset string is passed, it represents the windows
            total temporal extent.

        min_periods : int
            Minimum number of valid timestamps that are necessary to be present in any window, in order to trigger condition testing for this window.
            Windows with fewer timestamps are skipped. Must be >= 2, because a single
            value is always considered constant.
        """
        d: pd.Series = self._data[field]

        # 1. find starting points of consecutive constant values as a boolean mask
        # 2. fill the whole window with True's
        rolling = d.rolling(window=window, min_periods=min_periods)
        starting_points_mask = rolling.max() - rolling.min() <= thresh

        starting_points_mask = removeRollingRamps(starting_points_mask, window=window)

        # mimic forward rolling by roll over inverse [::-1]

        rolling = starting_points_mask[::-1].rolling(window=window, min_periods=0)
        # mimic any()
        mask = (rolling.sum()[::-1] > 0) & d.notna()

        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagByVariance(
        self: SaQC,
        field: str,
        window: OffsetStr,
        thresh: Float >= 0,
        maxna: (Int >= 0) | None = None,
        maxna_group: (Int >= 0) | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
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

        window = pd.Timedelta(window)

        delta = getFreqDelta(d.index)
        if not delta:
            raise IndexError("Timeseries irregularly sampled!")

        if maxna is None:
            maxna = np.inf
        if maxna_group is None:
            maxna_group = np.inf

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
