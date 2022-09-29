#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from saqc.constants import BAD
from saqc.core.register import flagging
from saqc.lib.tools import statPass

if TYPE_CHECKING:
    from saqc.core.core import SaQC


class NoiseMixin:
    @flagging()
    def flagByStatLowPass(
        self: "SaQC",
        field: str,
        func: Callable[[np.ndarray, pd.Series], float],
        window: str | pd.Timedelta,
        thresh: float,
        sub_window: str | pd.Timedelta | None = None,
        sub_thresh: float | None = None,
        min_periods: int | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag *chunks* of length, `window`:

        1. If they excexceed `thresh` with regard to `stat`:
        2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_window`,
           `excexceed `sub_thresh` with regard to `stat`:

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        func: Callable[[np.array, pd.Series], float]
            Function to aggregate chunk contnent with.

        window: str
            Temporal extension of the chunks to test

        thresh: float
            Threshold, that triggers flagging, if exceeded by stat value.

        sub_window: str, default None,
            Window size of the sub chunks, that are additionally tested for exceeding
            `sub_thresh` with respect to `stat`.

        sub_thresh: float, default None

        min_periods: int, default None

        flag : float, default BAD
            flag to set

        Returns
        -------
        saqc.SaQC
        """

        datcol = self._data[field]
        if not min_periods:
            min_periods = 0
        if not sub_thresh:
            sub_thresh = thresh
        window = pd.Timedelta(window)

        if sub_window is not None:
            sub_window = pd.Timedelta(sub_window)

        to_set = statPass(
            datcol,
            func,
            window,
            thresh,
            operator.gt,
            sub_window,
            sub_thresh,
            min_periods,
        )
        self._flags[to_set, field] = flag
        return self
