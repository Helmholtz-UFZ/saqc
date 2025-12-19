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
from saqc.lib.types import (
    Float,
    Int,
    OffsetStr,
    SaQC,
    SaQCFields,
    ValidatePublicMembers,
)
from saqc.parsing.environ import ENV_OPERATORS


class NoiseMixin(ValidatePublicMembers):

    @flagging()
    def flagByScatterLowpass(
        self: SaQC,
        field: SaQCFields,
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
        Flag data noisy data.

        Breaks up data in chunks and flags those chunks, if data there is too scattered.
        See notes section for algorithm details.

        Parameters
        ----------
        func :
            Scatter statistic.

            A function that assigns each data chunk its scattering.
            * ``"std"`` — standard deviation
            * ``"var"`` — variance
            * ``"mad"`` — median absolute deviation
            * ``Callable`` — custom function mapping 1D arrays to scalars.

        window :
            Scatter context size.

            The extension of the window, the scattering (usually variance) will be computed from, for each period.

        thresh :
            Scattering upper bound.

            If the scatter statistic obtained from a window of size `window` exceeds `thresh`, the value centered
            in the window is flagged.

        sub_window :
            Size of partitions of the scatter context.

            The window determining the context for the scatter statistics calculation is divided up into disjoint
            sub windows of size ``sub_window``, where the scattering is tested to exceed `sub_thresh`, in order to finally
            trigger flagging.

        sub_thresh :
            Scattering upper bound on sub window.

            Threshold, the statistic on every sub chunk is checked against. ``func(sub_chunk) > sub_thresh``.

        min_periods :
            Minimum window population.

            Ignored if ``window`` is an integer.

        Notes
        -----
        Chunks of length ``window`` are flagged if:

        1. They exceed ``thresh`` according to the function ``func``.
        2. All (possibly overlapping) sub-chunks of length ``sub_window`` exceed ``sub_thresh``
           according to the same function.

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
