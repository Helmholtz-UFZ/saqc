#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Callable, Any, Optional
from typing_extensions import Literal

import numpy as np
from scipy.spatial.distance import pdist

from saqc.core.modules.base import ModuleBase


class Drift(ModuleBase):
    def flagDriftFromNorm(
            self,
            field: str,
            fields: Sequence[str],
            segment_freq: str,
            norm_spread: float,
            norm_frac: float=0.5,
            metric: Callable[[np.array, np.array], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            linkage_method: Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]="single",
            **kwargs
    ):
        return self.defer("flagDriftFromNorm", locals())

    def flagDriftFromReference(
            self,
            field: str,
            fields: Sequence[str],
            segment_freq: str,
            thresh: float,
            metric: Callable[[np.array, np.array], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            **kwargs
    ):
        return self.defer("flagDriftFromReference", locals())

    def flagDriftFromScaledNorm(
            self,
            field: str,
            fields_scale1: Sequence[str],
            fields_scale2: Sequence[str],
            segment_freq: str,
            norm_spread: float,
            norm_frac: float=0.5,
            metric: Callable[[np.array, np.array], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            linkage_method: Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]="single",
            **kwargs
    ):
        return self.defer("flagDriftFromScaledNorm", locals())

    def correctExponentialDrift(
            self,
            field: str,
            maint_data_field: str,
            cal_mean: int = 5,
            flag_maint_period: bool = False,
            **kwargs
    ):
        return self.defer("correctExponentialDrift", locals())

    def correctRegimeAnomaly(
            self,
            field: str,
            cluster_field: str,
            model: Callable[[np.array, Any], np.array],
            regime_transmission: Optional[str] = None,
            x_date: bool = False
    ):
        return self.defer("correctRegimeAnomaly", locals())

    def correctOffset(
            self,
            field: str,
            max_mean_jump: float,
            normal_spread: float,
            search_winsz: str,
            min_periods: int,
            regime_transmission: Optional[str] = None,
            **kwargs
    ):
        return self.defer("correctOffset", locals())
