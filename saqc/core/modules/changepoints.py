#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple

import numpy as np
from dios import DictOfSeries
from typing_extensions import Literal

from saqc.constants import *
from saqc.core import Flags as Flagger
from saqc.core.modules.base import ModuleBase
from saqc.lib.types import FreqString, IntegerWindow


class ChangePoints(ModuleBase):

    def flagChangePoints(
            self, field: str, 
            stat_func: Callable[[np.ndarray, np.ndarray], float],
            thresh_func: Callable[[np.ndarray, np.ndarray], float],
            bwd_window: FreqString,
            min_periods_bwd: IntegerWindow,
            fwd_window: Optional[FreqString] = None,
            min_periods_fwd: Optional[IntegerWindow] = None,
            closed: Literal["right", "left", "both", "neither"] = "both",
            try_to_jit: bool = True,  # TODO rm, not a user decision
            reduce_window: FreqString = None,
            reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: x.argmax(),
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagChangePoints", locals())

    def assignChangePointCluster(
            self, field: str, 
            stat_func: Callable[[np.array, np.array], float],
            thresh_func: Callable[[np.array, np.array], float],
            bwd_window: str,
            min_periods_bwd: int,
            fwd_window: str = None,
            min_periods_fwd: Optional[int] = None,
            closed: Literal["right", "left", "both", "neither"] = "both",
            try_to_jit: bool = True,  # TODO: rm, not a user decision
            reduce_window: str = None,
            reduce_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, _: x.argmax(),
            model_by_resids: bool = False,
            flag_changepoints: bool = False,
            assign_cluster: bool = True,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("assignChangePointCluster", locals())
