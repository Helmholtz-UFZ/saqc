#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Callable, Optional, Union
from typing_extensions import Literal

import numpy as np
from scipy.spatial.distance import pdist

from saqc.constants import BAD
import saqc
from saqc.funcs import LinkageString
from saqc.lib.types import CurveFitter
from sphinxdoc.scripts.templates import doc

class Drift:

    @doc(saqc.drift.flagDriftFromNorm.__doc__)
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
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagDriftFromNorm", locals())

    @doc(saqc.drift.flagDriftFromReference.__doc__)
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
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagDriftFromReference", locals())


    @doc(saqc.drift.correctDrift.__doc__)
    def correctDrift(
        self,
        field: str,
        maintenance_field: str,
        model: Callable[..., float] | Literal["linear", "exponential"],
        cal_range: int = 5,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctDrift", locals())


    @doc(saqc.drift.correctRegimeAnomaly.__doc__)
    def correctRegimeAnomaly(
        self,
        field: str,
        cluster_field: str,
        model: CurveFitter,
        tolerance: Optional[str] = None,
        epoch: bool = False,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctRegimeAnomaly", locals())


    @doc(saqc.drift.correctOffset.__doc__)
    def correctOffset(
        self,
        field: str,
        max_jump: float,
        spread: float,
        window: str,
        min_periods: int,
        tolerance: Optional[str] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctOffset", locals())
