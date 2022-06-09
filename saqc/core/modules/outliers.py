#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.lib.docurator import doc


class Outliers:
    @doc(saqc.funcs.outliers.flagByStray.__doc__)
    def flagByStray(
        self,
        field: str,
        window: Optional[Union[int, str]] = None,
        min_periods: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByStray", locals())

    @doc(saqc.funcs.outliers.flagMVScores.__doc__)
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

    @doc(saqc.funcs.outliers.flagRaise.__doc__)
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

    @doc(saqc.funcs.outliers.flagMAD.__doc__)
    def flagMAD(
        self,
        field: str,
        window: str,
        z: float = 3.5,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagMAD", locals())

    @doc(saqc.funcs.outliers.flagOffset.__doc__)
    def flagOffset(
        self,
        field: str,
        tolerance: float,
        window: Union[int, str],
        thresh: Optional[float] = None,
        thresh_relative: Optional[float] = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagOffset", locals())

    @doc(saqc.funcs.outliers.flagByGrubbs.__doc__)
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

    @doc(saqc.funcs.outliers.flagRange.__doc__)
    def flagRange(
        self,
        field: str,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagRange", locals())

    @doc(saqc.funcs.outliers.flagCrossStatistics.__doc__)
    def flagCrossStatistics(
        self,
        field: Sequence[str],
        thresh: float,
        method: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagCrossStatistics", locals())
