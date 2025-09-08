#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np
import pandas as pd

from saqc.constants import BAD
from saqc.core.register import flagging
from saqc.lib.tools import isunflagged, statPass
from saqc.lib.types import Float, Int, OffsetStr, SaQC, ValidatePublicMembers
from saqc.parsing.environ import ENV_OPERATORS


class NoiseMixin(ValidatePublicMembers):

    @flagging()
    def flagByScatterLowpass(
        self: SaQC,
        field: str,
        window: OffsetStr | pd.Timedelta,
        thresh: Float >= 0,
        func: (
            Literal["std", "var", "mad"] | Callable[[np.ndarray, pd.Series], float]
        ) = "std",
        sub_window: OffsetStr | pd.Timedelta | None = None,
        sub_thresh: (Float >= 0) | None = None,
        min_periods: (Int >= 0) | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Flag anomalous data chunks based on scatter statistics.

        Chunks of length ``window`` are flagged if:

        1. They exceed ``thresh`` according to the function ``func``.
        2. All (possibly overlapping) sub-chunks of length ``sub_window`` exceed ``sub_thresh``
           according to the same function.

        Parameters
        ----------
        func : {"std", "var", "mad"} or Callable[[np.ndarray, pd.Series], float]
            Function to compute deviation for each chunk:
            * ``"std"`` — standard deviation
            * ``"var"`` — variance
            * ``"mad"`` — median absolute deviation
            * Callable — custom function mapping 1D arrays to scalars.

        window : str or pandas.Timedelta
            Size of the main chunk (time-based).

        thresh : float
            Threshold, the statistic of the main chunk is checked against. ``func(chunk) > thresh``.

        sub_window : str or pandas.Timedelta, optional
            Size of sub-chunks for secondary testing.

        sub_thresh : float, optional
            Threshold, the statistic of the main chunk is checked against. ``func(sub_chunk) > sub_thresh``.

        min_periods : int, optional
            Minimum number of values required in a chunk to perform the test.
            Ignored if ``window`` is an integer.
        """
        if sub_window is not None:
            sub_window = pd.Timedelta(sub_window)

        if isinstance(func, str):
            func = ENV_OPERATORS[func]

        to_set = statPass(
            datcol=self._data[field],
            stat=func,
            winsz=pd.Timedelta(window),
            thresh=thresh,
            comparator=operator.gt,
            sub_winsz=sub_window,
            sub_thresh=sub_thresh or thresh,
            min_periods=min_periods or 0,
        )
        mask = isunflagged(self._flags[field], kwargs["dfilter"]) & to_set
        self._flags[mask, field] = flag
        return self
