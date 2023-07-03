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
from saqc.lib.checking import validateCallable, validateMinPeriods, validateWindow
from saqc.lib.tools import isunflagged, statPass

if TYPE_CHECKING:
    from saqc import SaQC


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
        Flag data chunks of length ``window``, if:

        1. they excexceed ``thresh`` with regard to ``func`` and
        2. all (maybe overlapping) sub-chunks of the data chunks with length ``sub_window``,
           exceed ``sub_thresh`` with regard to ``func``

        Parameters
        ----------
        func :
            Aggregation function applied on every chunk.

        window :
            Window (i.e. chunk) size.

        thresh :
            Threshold. A given chunk is flagged, if the return value of ``func`` excceeds ``thresh``.

        sub_window :
            Window size of sub chunks, that are additionally tested for exceeding ``sub_thresh``
            with respect to ``func``.

        sub_thresh :
            Threshold. A given sub chunk is flagged, if the return value of ``func` excceeds ``sub_thresh``.

        min_periods :
            Minimum number of values needed in a chunk to perfom the test.
            Ignored if ``window`` is an integer.
        """
        validateCallable(func, "func")
        validateWindow(window, allow_int=False)
        validateMinPeriods(min_periods)
        if sub_window is not None:
            validateWindow(sub_window, "sub_window", allow_int=False)
            sub_window = pd.Timedelta(sub_window)

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
