#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.lib.docurator import doc


class ChangePoints:
    @doc(saqc.funcs.changepoints.flagChangePoints.__doc__)
    def flagChangePoints(
        self,
        field: str,
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        window: str | Tuple[str, str],
        min_periods: int | Tuple[int, int],
        closed: Literal["right", "left", "both", "neither"] = "both",
        reduce_window: str = None,
        reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: x.argmax(),
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagChangePoints", locals())

    @doc(saqc.funcs.changepoints.assignChangePointCluster.__doc__)
    def assignChangePointCluster(
        self,
        field: str,
        stat_func: Callable[[np.array, np.array], float],
        thresh_func: Callable[[np.array, np.array], float],
        window: str | Tuple[str, str],
        min_periods: int | Tuple[int, int],
        closed: Literal["right", "left", "both", "neither"] = "both",
        reduce_window: str = None,
        reduce_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = lambda x, _: x.argmax(),
        model_by_resids: bool = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("assignChangePointCluster", locals())
