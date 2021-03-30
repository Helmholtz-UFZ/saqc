#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Callable, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist

from saqc.constants import *
from saqc.core.modules.base import ModuleBase
from saqc.funcs import LinkageString, DictOfSeries, Flags
from saqc.lib.types import ColumnName, FreqString, CurveFitter


class Drift(ModuleBase):
    def flagDriftFromNorm(
            self, 
            field: ColumnName,
            fields: Sequence[ColumnName],
            segment_freq: FreqString,
            norm_spread: float,
            norm_frac: float = 0.5,
            metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            linkage_method: LinkageString = "single",
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagDriftFromNorm", locals())

    def flagDriftFromReference(
            self, 
            field: ColumnName,
            fields: Sequence[ColumnName],
            segment_freq: FreqString,
            thresh: float,
            metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagDriftFromReference", locals())

    def flagDriftFromScaledNorm(
            self, 
            field: ColumnName,
            fields_scale1: Sequence[ColumnName],
            fields_scale2: Sequence[ColumnName],
            segment_freq: FreqString,
            norm_spread: float,
            norm_frac: float = 0.5,
            metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
            linkage_method: LinkageString = "single",
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagDriftFromScaledNorm", locals())

    def correctExponentialDrift(
            self, 
            field: ColumnName,
            maint_data_field: ColumnName,
            cal_mean: int = 5,
            flag_maint_period: bool = False,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("correctExponentialDrift", locals())

    def correctRegimeAnomaly(
            self, 
            field: ColumnName,
            cluster_field: ColumnName,
            model: CurveFitter,
            regime_transmission: Optional[FreqString] = None,
            x_date: bool = False,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
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
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("correctOffset", locals())
