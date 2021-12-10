#! /usr/bin/env python
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


class Drift:
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

    def correctDrift(
        self,
        field: str,
        maintenance_field: str,
        model: Callable[..., float] | Literal["linear", "exponential"],
        cal_range: int = 5,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("correctDrift", locals())

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
