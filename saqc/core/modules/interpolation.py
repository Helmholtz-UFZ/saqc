#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Union, Callable

import numpy as np
import pandas as pd

from saqc.constants import UNFLAGGED
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.funcs.interpolation import _SUPPORTED_METHODS


class Interpolation(ModuleBase):
    def interpolateByRolling(
        self,
        field: str,
        winsz: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs
    ) -> saqc.SaQC:
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
    ) -> saqc.SaQC:
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
    ) -> saqc.SaQC:
        return self.defer("interpolateIndex", locals())
