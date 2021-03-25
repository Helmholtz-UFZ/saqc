#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Callable, Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dios import DictOfSeries
from typing_extensions import Literal

from saqc.core import Flags as Flagger
from saqc.constants import *
from saqc.core.modules.base import ModuleBase
from saqc.funcs.interpolation import _SUPPORTED_METHODS


class Interpolation(ModuleBase):

    def interpolateByRolling(
            self, field: str, 
            winsz: Union[str, int],
            func: Callable[[pd.Series], float] = np.median,
            center: bool = True,
            min_periods: int = 0,
            flag: float = UNFLAGGED,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("interpolateByRolling", locals())

    def interpolateInvalid(
            self, 
            field: str,
            method: _SUPPORTED_METHODS,
            inter_order: int = 2,
            inter_limit: int = 2,
            downgrade_interpolation: bool = False,
            flag: float = UNFLAGGED,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("interpolateInvalid", locals())

    def interpolateIndex(
            self, 
            field: str,
            freq: str,
            method: _SUPPORTED_METHODS,
            inter_order: int = 2,
            inter_limit: int = 2,
            downgrade_interpolation: bool = False,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("interpolateIndex", locals())

