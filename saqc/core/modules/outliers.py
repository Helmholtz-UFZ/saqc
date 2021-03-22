#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Callable, Sequence
from typing_extensions import Literal

import numpy as np
import pandas as pd

from saqc.core.modules.base import ModuleBase


class Outliers(ModuleBase):

    def flagByStray(
            self,
            field: str,
            partition_freq: Optional[Union[str, int]] = None,
            partition_min: int = 11,
            iter_start: float = 0.5,
            alpha: float = 0.05,
            **kwargs
    ):
        return self.defer("flagByStray", locals())

    def flagMVScores(
            self,
            field: str,
            fields: Sequence[str],
            trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
            alpha: float = 0.05,
            n_neighbors: int = 10,
            scoring_func: Callable[[pd.Series], float] = np.sum,
            iter_start: float = 0.5,
            stray_partition: Optional[Union[str, int]] = None,
            stray_partition_min: int = 11,
            trafo_on_partition: bool = True,
            reduction_range: Optional[str] = None,
            reduction_drop_flagged: bool = False,
            reduction_thresh: float = 3.5,
            reduction_min_periods: int = 1,
            **kwargs,
    ):
        return self.defer("flagMVScores", locals())

    def flagRaise(
            self,
            field: str,
            thresh: float,
            raise_window: str,
            intended_freq: str,
            average_window: Optional[str] = None,
            mean_raise_factor: float = 2.,
            min_slope: Optional[float] = None,
            min_slope_weight: float = 0.8,
            numba_boost: bool = True,
            **kwargs,
    ):
        return self.defer("flagRaise", locals())

    def flagMAD(self, field: str, window: str, z: float = 3.5, **kwargs):
        return self.defer("flagMAD", locals())

    def flagOffset(
            self,
            field: str,
            thresh: float,
            tolerance: float,
            window: str,
            rel_thresh: Optional[float]=None,
            numba_kickin: int = 200000,
            **kwargs
    ):
        return self.defer("flagOffset", locals())

    def flagByGrubbs(
            self,
            field: str,
            winsz: Union[str, int],
            alpha: float = 0.05,
            min_periods: int = 8,
            check_lagged: bool = False,
            **kwargs
    ):
        return self.defer("flagByGrubbs", locals())

    def flagRange(
            self,
            field: str,
            min: float = -np.inf,
            max: float = np.inf,
            **kwargs
    ):
        return self.defer("flagRange", locals())

    def flagCrossStatistic(
            self,
            field: str,
            fields: Sequence[str],
            thresh: float,
            cross_stat: Literal["modZscore", "Zscore"] = "modZscore",
            **kwargs
    ):
        return self.defer("flagCrossStatistic", locals())
