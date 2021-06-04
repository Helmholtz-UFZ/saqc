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
        segment_freq: FreqString,
        norm_spread: float,
        norm_frac: float = 0.5,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        linkage_method: LinkageString = "single",
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagDriftFromNorm", locals())

    def flagDriftFromReference(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        segment_freq: FreqString,
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
        fields_scale1: Sequence[ColumnName],
        fields_scale2: Sequence[ColumnName],
        segment_freq: FreqString,
        norm_spread: float,
        norm_frac: float = 0.5,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        linkage_method: LinkageString = "single",
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagDriftFromScaledNorm", locals())

    def correctDrift(
        self,
        field: ColumnName,
        maint_data_field: ColumnName,
        driftModel: Callable[..., float],
        cal_mean: int = 5,
        flag_maint_period: bool = False,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctDrift", locals())

    def correctRegimeAnomaly(
        self,
        field: ColumnName,
        cluster_field: ColumnName,
        model: CurveFitter,
        regime_transmission: Optional[FreqString] = None,
        x_date: bool = False,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctRegimeAnomaly", locals())

    def correctOffset(
        self,
        field: ColumnName,
        max_mean_jump: float,
        normal_spread: float,
        search_winsz: FreqString,
        min_periods: int,
        regime_transmission: Optional[FreqString] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("correctOffset", locals())
