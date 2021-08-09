#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Callable, Union, Literal

import numpy as np
import pandas as pd

from saqc.core import Flags
from saqc.core.modules.base import ModuleBase
import saqc


class Scores(ModuleBase):
    def assignKNNScore(
        self,
        field: str,
        fields: Sequence[str],
        n_neighbors: int = 10,
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        trafo_on_partition: bool = True,
        scoring_func: Callable[[pd.Series], float] = np.sum,
        target_field: str = "kNN_scores",
        partition_freq: Union[float, str] = np.inf,
        partition_min: int = 2,
        kNN_algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        metric: str = "minkowski",
        p: int = 2,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("assignKNNScore", locals())
