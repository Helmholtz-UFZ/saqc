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
    def flagByStatLowPass(
        self: SaQC,
        field: str,
        window: OffsetStr,
        thresh: Float >= 0,
        func: (
            Literal["std", "var", "mad"] | Callable[[np.ndarray, pd.Series], float]
        ) = "std",
        sub_window: (Float > 0) | OffsetStr | None = None,
        sub_thresh: (Float > 0) | None = None,
        min_periods: (Int >= 0) | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Flag data chunks of length ``window`` dependent on the data deviation.

        Flag data chunks of length ``window`` if

        1. they excexceed ``thresh`` with regard to ``func`` and
        2. all (maybe overlapping) sub-chunks of the data chunks with length ``sub_window``,
           exceed ``sub_thresh`` with regard to ``func``

            .. deprecated:: 2.5.0
               Deprecated Function. See :py:meth:`~saqc.SaQC.flagByScatterLowpass`.

        Parameters
        ----------
        func :
            Either a String value, determining the aggregation function applied on every chunk.

            * 'std': standard deviation
            * 'var': variance
            * 'mad': median absolute deviation

            Or a Callable function mapping 1 dimensional arraylikes onto scalars.

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
        warnings.warn(
            "function 'flagByStatLowPass' is deprecated and will be removed in a future release, "
            "use 'flagByScatterLowpass' instead.",
            DeprecationWarning,
        )

        return self.flagByScatterLowpass(
            field=field,
            window=window,
            thresh=thresh,
            func=func,
            sub_window=sub_window,
            sub_thresh=sub_thresh,
            min_periods=min_periods,
            flag=flag,
        )

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
        Flag data chunks of length ``window`` dependent on the data deviation.

        Flag data chunks of length ``window`` if

        1. they excexceed ``thresh`` with regard to ``func`` and
        2. all (maybe overlapping) sub-chunks of the data chunks with length ``sub_window``,
           exceed ``sub_thresh`` with regard to ``func``

        Parameters
        ----------
        func :
            Either a string, determining the aggregation function applied on every chunk:

            * 'std': standard deviation
            * 'var': variance
            * 'mad': median absolute deviation

            Or a Callable, mapping 1 dimensional array likes onto scalars.

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
