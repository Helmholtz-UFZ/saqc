#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.lib.types import IntegerWindow, FreqString, ColumnName


class Outliers(ModuleBase):
    def flagByStray(
        self,
        field: ColumnName,
        freq: Optional[Union[IntegerWindow, FreqString]] = None,
        min_periods: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagByStray", locals())

    def flagMVScores(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        iter_start: float = 0.5,
        partition: Optional[Union[IntegerWindow, FreqString]] = None,
        partition_min: int = 11,
        partition_trafo: bool = True,
        stray_range: Optional[FreqString] = None,
        drop_flagged: bool = False,  # TODO: still a case ?
        thresh: float = 3.5,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagMVScores", locals())

    def flagRaise(
        self,
        field: ColumnName,
        thresh: float,
        raise_window: FreqString,
        freq: FreqString,
        average_window: Optional[FreqString] = None,
        raise_factor: float = 2.0,
        slope: Optional[float] = None,
        weight: float = 0.8,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagRaise", locals())

    def flagMAD(
        self,
        field: ColumnName,
        window: FreqString,
        z: float = 3.5,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagMAD", locals())

    def flagOffset(
        self,
        field: ColumnName,
        thresh: float,
        tolerance: float,
        window: Union[IntegerWindow, FreqString],
        thresh_relative: Optional[float] = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagOffset", locals())

    def flagByGrubbs(
        self,
        field: ColumnName,
        window: Union[FreqString, IntegerWindow],
        alpha: float = 0.05,
        min_periods: int = 8,
        pedantic: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagByGrubbs", locals())

    def flagRange(
        self,
        field: ColumnName,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagRange", locals())

    def flagCrossStatistic(
        self,
        field: ColumnName,
        fields: Sequence[ColumnName],
        thresh: float,
        method: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagCrossStatistic", locals())
