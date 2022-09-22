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

from saqc.constants import BAD
from saqc.core.register import flagging
from saqc.lib.tools import customRoller, getFreqDelta, statPass
from saqc.lib.ts_operators import varQC

if TYPE_CHECKING:
    from saqc.core.core import SaQC


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
        field : str
            A column in flags and data.

        thresh : float
            Maximum total change allowed per window.

        window : str | int
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window.
            Each window will be a variable sized based on the observations included
            in the time-period.

        flag : float, default BAD
            Flag to set.

        Returns
        -------
        saqc.SaQC
        """
        if not isinstance(window, (str, int)):
            raise TypeError("window must be offset string or int.")

        d = self._data[field]

        # min_periods=2 ensures that at least two non-nan values are present
        # in each window and also min() == max() == d[i] is not possible.
        kws = dict(window=window, min_periods=min_periods, expand=False)

        # 1. find starting points of consecutive constant values as a boolean mask
        # 2. fill the whole window with True's
        rolling = customRoller(d, **kws)
        starting_points_mask = rolling.max() - rolling.min() <= thresh
        rolling = customRoller(starting_points_mask, **kws, forward=True)
        # mimic any()
        mask = (rolling.sum() > 0) & d.notna()

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
        field : str
            A column in flags and data.

        window : str | int
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window.
            Each window will be sized, based on the number of observations included
            in the time-period.

        thresh : float, default 0.0005
            Maximum total variance allowed per window.

        maxna : int, default None
            Maximum number of NaNs allowed in window.
            If more NaNs are present, the window is not flagged.

        maxna_group : int, default None
            Same as `maxna` but for consecutive NaNs.

        flag : float, default BAD
            Flag to set.

        Returns
        -------
        saqc.SaQC
        """
        dataseries = self._data[field]
        delta = getFreqDelta(dataseries.index)
        if not delta:
            raise IndexError("Timeseries irregularly sampled!")

        if maxna is None:
            maxna = np.inf

        if maxna_group is None:
            maxna_group = np.inf

        min_periods = int(np.ceil(pd.Timedelta(window) / pd.Timedelta(delta)))
        window = pd.Timedelta(window)
        to_set = statPass(
            dataseries,
            lambda x: varQC(x, maxna, maxna_group),
            window,
            thresh,
            operator.lt,
            min_periods=min_periods,
        )

        self._flags[to_set, field] = flag
        return self
