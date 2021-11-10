#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Callable, Optional

import numpy as np
from scipy.spatial.distance import pdist

from saqc.constants import BAD
import saqc
from saqc.funcs import LinkageString
from saqc.lib.types import FreqString, CurveFitter


class Drift:
    def flagDriftFromNorm(
        self,
        field: str,
        fields: Sequence[str],
        freq: FreqString,
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

    def flagDriftFromReference(
        self,
        field: str,
        fields: Sequence[str],
        freq: FreqString,
        thresh: float,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagDriftFromReference", locals())

    def flagDriftFromScaledNorm(
        self,
        field: str,
        set_1: Sequence[str],
        set_2: Sequence[str],
        freq: FreqString,
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
        return self._defer("flagDriftFromScaledNorm", locals())

    def correctDrift(
        self,
        field: str,
        maintenance_field: str,
        model: Callable[..., float],
        cal_range: int = 5,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctDrift", locals())

    def correctRegimeAnomaly(
        self,
        field: str,
        cluster_field: str,
        model: CurveFitter,
        tolerance: Optional[FreqString] = None,
        epoch: bool = False,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctRegimeAnomaly", locals())

    def correctOffset(
        self,
        field: str,
        max_jump: float,
        spread: float,
        window: FreqString,
        min_periods: int,
        tolerance: Optional[FreqString] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctOffset", locals())
