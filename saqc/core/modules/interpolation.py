#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Callable, Any, Optional, Sequence
from typing_extensions import Literal

import numpy as np
import pandas as pd

from saqc.common import *
from saqc.core.modules.base import ModuleBase


class Interpolation(ModuleBase):

    def interpolateByRolling(
            self,
            field: str,
            winsz: Union[str, int],
            func: Callable[[pd.Series], pd.Series] = np.median,
            center: bool = True,
            min_periods: int = 0,
            interpol_flag=Any,
            **kwargs
    ):
        return self.defer("interpolateByRolling", locals())

    def interpolateInvalid(
            self,
            field: str,
            method: Literal["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"],
            inter_order: int = 2,
            inter_limit: int = 2,
            interpol_flag: float = UNFLAGGED,
            downgrade_interpolation: bool = False,
            not_interpol_flags: Optional[Union[float, Sequence[float]]] = None,
            **kwargs
    ):
        return self.defer("interpolateInvalid", locals())

    def interpolateIndex(
            self,
            field: str,
            freq: str,
            method: Literal["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"],
            inter_order: int = 2,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            downgrade_interpolation: bool = False,
            empty_intervals_flag: Any = None,
            grid_field: str = None,
            inter_limit: int = 2,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ):
        return self.defer("interpolateIndex", locals())

    def interpolateInvalid(
            self,
            field: str,
            method: Literal["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"],
            inter_order: int = 2,
            inter_limit: int = 2,
            interpol_flag: float = UNFLAGGED,
            downgrade_interpolation: bool = False,
            not_interpol_flags: Optional[Union[Any, Sequence[Any]]] = None,
            **kwargs
    ):
        return self.defer("interpolateInvalid", locals())

    def interpolateIndex(
            self,
            field: str,
            freq: str,
            method: Literal["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"],
            inter_order: int = 2,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            downgrade_interpolation: bool = False,
            empty_intervals_flag: Any = None,
            grid_field: str = None,
            inter_limit: int = 2,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ):
        return self.defer("interpolateIndex", locals())
