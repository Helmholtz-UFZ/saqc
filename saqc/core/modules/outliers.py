#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
from saqc.lib.types import IntegerWindow, FreqString, ColumnName


class Outliers(ModuleBase):
    def flagByStray(
        self,
        field: ColumnName,
        partition_freq: Optional[Union[IntegerWindow, FreqString]] = None,
        partition_min: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagByStray", locals())

    def flagMVScores(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n_neighbors: int = 10,
        scoring_func: Callable[[pd.Series], float] = np.sum,
        iter_start: float = 0.5,
        stray_partition: Optional[Union[IntegerWindow, FreqString]] = None,
        stray_partition_min: int = 11,
        trafo_on_partition: bool = True,
        reduction_range: Optional[FreqString] = None,
        reduction_drop_flagged: bool = False,  # TODO: still a case ?
        reduction_thresh: float = 3.5,
        reduction_min_periods: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagMVScores", locals())

    def flagRaise(
        self,
        field: ColumnName,
        thresh: float,
        raise_window: FreqString,
        intended_freq: FreqString,
        average_window: Optional[FreqString] = None,
        mean_raise_factor: float = 2.0,
        min_slope: Optional[float] = None,
        min_slope_weight: float = 0.8,
        numba_boost: bool = True,  # TODO: rm, not a user decision
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagRaise", locals())

    def flagMAD(
        self,
        field: ColumnName,
        window: FreqString,
        z: float = 3.5,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagMAD", locals())

    def flagOffset(
        self,
        field: ColumnName,
        thresh: float,
        tolerance: float,
        window: Union[IntegerWindow, FreqString],
        rel_thresh: Optional[float] = None,
        numba_kickin: int = 200000,  # TODO: rm, not a user decision
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagOffset", locals())

    def flagByGrubbs(
        self,
        field: ColumnName,
        winsz: Union[FreqString, IntegerWindow],
        alpha: float = 0.05,
        min_periods: int = 8,
        check_lagged: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagByGrubbs", locals())

    def flagRange(
        self,
        field: ColumnName,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagRange", locals())

    def flagCrossStatistic(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        thresh: float,
        cross_stat: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        return self.defer("flagCrossStatistic", locals())
