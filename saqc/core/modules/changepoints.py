#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from typing_extensions import Literal

from saqc.constants import BAD
import saqc
from saqc.lib.types import FreqString


class ChangePoints:
    def flagChangePoints(
        self,
        field: str,
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        window: FreqString | Tuple[FreqString, FreqString],
        min_periods: int | Tuple[int, int],
        closed: Literal["right", "left", "both", "neither"] = "both",
        reduce_window: FreqString = None,
        reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: x.argmax(),
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagChangePoints", locals())

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
        set_flags: bool = False,
        assign_cluster: bool = True,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("assignChangePointCluster", locals())
