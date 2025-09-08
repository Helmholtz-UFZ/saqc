#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-


"""
Detecting breaks in data.

This module provides functions to detect and flag breaks in data, for example temporal
gaps (:py:func:`flagMissing`), jumps and drops (:py:func:`flagJumps`) or temporal
isolated values (:py:func:`flagIsolated`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from saqc import BAD, FILTER_ALL
from saqc.core import flagging, register
from saqc.funcs.changepoints import _getChangePoints
from saqc.lib.tools import isunflagged
from saqc.lib.types import Float, Int, OffsetStr, SaQC, ValidatePublicMembers


class BreaksMixin(ValidatePublicMembers):

    @register(mask=[], demask=[], squeeze=["field"])
    def flagMissing(
        self: SaQC,
        field: str,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> SaQC:
        """
        .. deprecated:: 2.7.0
           Deprecated Function. Please use to :py:meth:`~saqc.SaQC.flagNaN` instead.
        """

        return self.flagNAN(field, flag, dfilter, **kwargs)

    @register(mask=[], demask=[], squeeze=["field"])
    def flagNAN(
        self: SaQC,
        field: str,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> SaQC:
        """
        Flag NaNs in data.

        By default, only NaNs are flagged, that not already have a flag.
        `dfilter` can be used to pass a flag that is used as threshold.
        Each flag worse than the threshold is replaced by the function.
        This is, because the data gets masked (with NaNs) before the
        function evaluates the NaNs.
        """

        datacol = self._data[field]
        mask = datacol.isna()

        mask = isunflagged(self._flags[field], dfilter) & mask

        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagIsolated(
        self: SaQC,
        field: str,
        gap_window: OffsetStr,
        group_window: OffsetStr,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Find and flag temporally isolated groups of data.

        The function flags groups of values that are surrounded by sufficiently large
        data gaps. A data gap is a timespan containing no valid data. (Data is valid if it is not `NaN` and if it is not assigned a flag with a level higher than the functions `flag` value).

        Parameters
        ----------
        gap_window : str
            Minimum gap size required before and after a data group to consider it
            isolated. See conditions (2) and (3) below.

        group_window : str
            Maximum size of a data chunk to consider it a candidate for an isolated group.
            Data chunks larger than this are ignored. This does not include the possible
            gaps surrounding it. See condition (1) below.

        Notes
        -----
        A series of values :math:`x_k, x_{k+1}, ..., x_{k+n}` with timestamps
        :math:`t_k, t_{k+1}, ..., t_{k+n}` is considered isolated if:

        1. :math:`t_{k+1} - t_n <` `group_window`
        2. No valid values in a succeeding period of `gap_window` extension.
        3. No valid values exist in the succeeding gap of size `gap_window`.
        """

        dat = self._data[field].dropna()
        if dat.empty:
            return self

        gap_ends = dat.rolling(gap_window).count() == 1
        gap_ends.iloc[0] = False
        gap_ends = gap_ends[gap_ends]
        gap_starts = dat[::-1].rolling(gap_window).count()[::-1] == 1
        gap_starts.iloc[-1] = False
        gap_starts = gap_starts[gap_starts]
        if gap_starts.empty:
            return self

        gap_starts = gap_starts[1:]
        gap_ends = gap_ends[:-1]
        isolated_groups = gap_starts.index - gap_ends.index < group_window
        gap_starts = gap_starts[isolated_groups]
        gap_ends = gap_ends[isolated_groups]
        to_flag = pd.Series(False, index=dat.index)
        for s, e in zip(gap_starts.index, gap_ends.index):
            # what gets flagged are the groups between the gaps, those range from
            # the end of one gap (gap_end) to the beginning of the next (gap_start)
            to_flag[e:s] = True

        to_flag = to_flag.reindex(self._data[field].index, fill_value=False)
        self._flags[to_flag.to_numpy(), field] = flag
        return self

    @flagging()
    def flagJumps(
        self: SaQC,
        field: str,
        thresh: Float >= 0,
        window: OffsetStr,
        min_periods: Int >= 0 = 0,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> SaQC:
        """
        Flag jumps and drops in data.

        Flags values where the mean changes significantly between two adjacent rolling
        windows, indicating a "jump" from one level to another. Whenever the difference
        between the means of the two windows exceeds `thresh`, the values between the
        windows are flagged.

        Parameters
        ----------
        thresh : float
            Threshold by which the mean of data must jump to trigger flagging.

        window : str
            Size of the two rolling windows. Determines the number of timestamps used
            for calculating the mean in each window. Windows should be chosen large enough to
            obtain a reliable mean. But not too large as well, since the window size implies a lower bound for the detection resolution.
            Jumps exceeding `thresh` but being apart from each other by less than 3/4 of the window size may not be detected reliably.

        min_periods : int
            Minimum number of timestamps in `window` required to calculate a valid mean. If no valid mean for the window can be calculated, flagging wont be triggered for the associated change point.

        Notes
        -----
        Jumps closer together than three fourths (3/4) of the window size may not be
        detected reliably.

        Examples
        --------
        Below diagram illustrates the interaction of parameters for a positive value jump
        initializing a new mean level.

        .. figure:: /resources/images/flagJumpsPic.png

           The two adjacent windows of size `window` roll through the data series. Whenever
           the mean values differ by more than `thresh`, flagging is triggered.
        """
        mask = _getChangePoints(
            data=self._data[field],
            stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
            thresh_func=lambda x, y: thresh,
            window=window,
            min_periods=min_periods,
            result="mask",
        )

        mask = isunflagged(self._flags[field], dfilter) & mask
        self._flags[mask, field] = flag
        return self
