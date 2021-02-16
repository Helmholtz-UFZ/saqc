#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union, Tuple
from typing_extensions import Literal

import numpy as np

from dios import DictOfSeries

from saqc.core.modules.base import ModuleBase
from saqc.flagger.baseflagger import BaseFlagger


class ChangePoints(ModuleBase):

    def flagChangePoints(
            self,
            field: str,
            stat_func: Callable[[np.array], np.array],
            thresh_func: Callable[[np.array], np.array],
            bwd_window: str,
            min_periods_bwd: Union[str, int],
            fwd_window: str = None,
            min_periods_fwd: Union[str, int] = None,
            closed: Literal["right", "left", "both", "neither"] = "both",
            try_to_jit: bool = True,
            reduce_window: str = None,
            reduce_func: Callable[[np.array, np.array], np.array] = lambda x, y: x.argmax(),
            **kwargs
    ) -> Tuple[DictOfSeries, BaseFlagger]:

        return self.defer("flagChangePoints", locals())

    def assignChangePointCluster(
            self,
            field: str,
            stat_func: Callable[[np.array], np.array],
            thresh_func: Callable[[np.array], np.array],
            bwd_window: str,
            min_periods_bwd: Union[str, int],
            fwd_window: str = None,
            min_periods_fwd: Union[str, int] = None,
            closed: Literal["right", "left", "both", "neither"] = "both",
            try_to_jit: bool = True,
            reduce_window: str = None,
            reduce_func: Callable[[np.array, np.array], np.array] = lambda x, y: x.argmax(),
            model_by_resids: bool = False,
            flag_changepoints: bool = False,
            assign_cluster: bool = True,
            **kwargs
    ) -> Tuple[DictOfSeries, BaseFlagger]:

        return self.defer("assignChangePointCluster", locals())
