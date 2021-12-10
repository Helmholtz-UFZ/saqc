#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
import saqc


class Outliers:
    def flagByStray(
        self,
        field: str,
        freq: Optional[Union[int, str]] = None,
        min_periods: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByStray", locals())

    def flagMVScores(
        self,
        field: Sequence[str],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        iter_start: float = 0.5,
        partition: Optional[Union[int, str]] = None,
        partition_min: int = 11,
        stray_range: Optional[str] = None,
        drop_flagged: bool = False,  # TODO: still a case ?
        thresh: float = 3.5,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagMVScores", locals())

    def flagRaise(
        self,
        field: str,
        thresh: float,
        raise_window: str,
        freq: str,
        average_window: Optional[str] = None,
        raise_factor: float = 2.0,
        slope: Optional[float] = None,
        weight: float = 0.8,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagRaise", locals())

    def flagMAD(
        self,
        field: str,
        window: str,
        z: float = 3.5,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagMAD", locals())

    def flagOffset(
        self,
        field: str,
        thresh: float,
        tolerance: float,
        window: Union[int, str],
        thresh_relative: Optional[float] = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagOffset", locals())

    def flagByGrubbs(
        self,
        field: str,
        window: Union[str, int],
        alpha: float = 0.05,
        min_periods: int = 8,
        pedantic: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByGrubbs", locals())

    def flagRange(
        self,
        field: str,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagRange", locals())

    def flagCrossStatistics(
        self,
        field: Sequence[str],
        thresh: float,
        method: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagCrossStatistic", locals())
