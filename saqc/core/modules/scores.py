#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Callable, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc


class Scores:
    def assignKNNScore(
        self,
        field: str,
        fields: Sequence[str],
        n: int = 10,
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        trafo_on_partition: bool = True,
        func: Callable[[pd.Series], float] = np.sum,
        target: str = "kNN_scores",
        freq: Union[float, str] = np.inf,
        min_periods: int = 2,
        method: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        metric: str = "minkowski",
        p: int = 2,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("assignKNNScore", locals())
