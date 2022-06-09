#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import pdist
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.funcs import LinkageString
from saqc.lib.docurator import doc
from saqc.lib.types import CurveFitter


class Drift:
    @doc(saqc.funcs.drift.flagDriftFromNorm.__doc__)
    def flagDriftFromNorm(
        self,
        field: Sequence[str],
        freq: str,
        spread: float,
        frac: float = 0.5,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        method: LinkageString = "single",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagDriftFromNorm", locals())

    @doc(saqc.funcs.drift.flagDriftFromReference.__doc__)
    def flagDriftFromReference(
        self,
        field: Sequence[str],
        reference: str,
        freq: str,
        thresh: float,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagDriftFromReference", locals())

    @doc(saqc.funcs.drift.correctDrift.__doc__)
    def correctDrift(
        self,
        field: str,
        maintenance_field: str,
        model: Callable[..., float] | Literal["linear", "exponential"],
        cal_range: int = 5,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("correctDrift", locals())

    @doc(saqc.funcs.drift.correctRegimeAnomaly.__doc__)
    def correctRegimeAnomaly(
        self,
        field: str,
        cluster_field: str,
        model: CurveFitter,
        tolerance: Optional[str] = None,
        epoch: bool = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("correctRegimeAnomaly", locals())

    @doc(saqc.funcs.drift.correctOffset.__doc__)
    def correctOffset(
        self,
        field: str,
        max_jump: float,
        spread: float,
        window: str,
        min_periods: int,
        tolerance: Optional[str] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("correctOffset", locals())

    @doc(saqc.funcs.drift.flagRegimeAnomaly.__doc__)
    def flagRegimeAnomaly(
        self,
        field: str,
        cluster_field: str,
        spread: float,
        method: LinkageString = "single",
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.abs(
            np.nanmean(x) - np.nanmean(y)
        ),
        frac: float = 0.5,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagRegimeAnomaly", locals())

    @doc(saqc.funcs.drift.assignRegimeAnomaly.__doc__)
    def assignRegimeAnomaly(
        self,
        field: str,
        cluster_field: str,
        spread: float,
        method: LinkageString = "single",
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.abs(
            np.nanmean(x) - np.nanmean(y)
        ),
        frac: float = 0.5,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("assignRegimeAnomaly", locals())
