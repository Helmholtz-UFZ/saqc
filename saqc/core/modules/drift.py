#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Callable, Optional

import numpy as np
from scipy.spatial.distance import pdist

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.funcs import LinkageString
from saqc.lib.types import ColumnName, FreqString, CurveFitter


class Drift(ModuleBase):
    def flagDriftFromNorm(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
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
        return self.defer("flagDriftFromNorm", locals())

    def flagDriftFromReference(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        freq: FreqString,
        thresh: float,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagDriftFromReference", locals())

    def flagDriftFromScaledNorm(
        self,
        field: ColumnName,
        set_1: Sequence[ColumnName],
        set_2: Sequence[ColumnName],
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
        return self.defer("flagDriftFromScaledNorm", locals())

    def correctDrift(
        self,
        field: ColumnName,
        maintenance_field: ColumnName,
        model: Callable[..., float],
        cal_range: int = 5,
        set_flags: bool = False,  # Todo: remove, user should use flagManual
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctDrift", locals())

    def correctRegimeAnomaly(
        self,
        field: ColumnName,
        cluster_field: ColumnName,
        model: CurveFitter,
        tolerance: Optional[FreqString] = None,
        epoch: bool = False,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctRegimeAnomaly", locals())

    def correctOffset(
        self,
        field: ColumnName,
        max_jump: float,
        spread: float,
        window: FreqString,
        min_periods: int,
        tolerance: Optional[FreqString] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctOffset", locals())
