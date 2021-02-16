#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Any, Sequence, Callable
from typing_extensions import Literal

import numpy as np
import pandas as pd

from saqc.core.modules.base import ModuleBase


class Resampling(ModuleBase):

    def aggregate(
            self,
            field: str,
            freq: str,
            value_func,
            flag_func: Callable[[pd.Series], pd.Series] = np.nanmax,
            method: Literal["fagg", "bagg", "nagg"] = "nagg",
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            **kwargs
    ):
        return self.defer("aggregate", locals())

    def linear(
            self,
            field: str,
            freq: str,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            **kwargs
    ):
        return self.defer("linear", locals())

    def interpolate(
            self,
            field: str,
            freq: str,
            method: Literal["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"],
            order: int = 1,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            **kwargs,
    ):
        return self.defer("interpolate", locals())

    def mapToOriginal(
            self,
            field: str,
            method: Literal["inverse_fagg", "inverse_bagg", "inverse_nagg", "inverse_fshift", "inverse_bshift", "inverse_nshift", "inverse_interpolation"],
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            **kwargs
    ):
        return self.defer("mapToOriginal", locals())

    def shift(
            self,
            field: str,
            freq: str,
            method: Literal["fshift", "bshift", "nshift"] = "nshift",
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            empty_intervals_flag: Optional[str] = None,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ):
        return self.defer("shift", locals())

    def resample(
            self,
            field: str,
            freq: str,
            agg_func: Callable[[pd.Series], pd.Series] = np.mean,
            method: Literal["fagg", "bagg", "nagg"] = "bagg",
            max_invalid_total_d: Optional[int] = None,
            max_invalid_consec_d: Optional[int] = None,
            max_invalid_consec_f: Optional[int] = None,
            max_invalid_total_f: Optional[int] = None,
            flag_agg_func: Callable[[pd.Series], pd.Series] = max,
            empty_intervals_flag: Optional[Any] = None,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            all_na_2_empty: bool = False,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ):
        return self.defer("resample", locals())

    def reindexFlags(
            self,
            field: str,
            method: Literal["inverse_fagg", "inverse_bagg", "inverse_nagg", "inverse_fshift", "inverse_bshift", "inverse_nshift"],
            source: str,
            freq: Optional[str] = None,
            to_drop: Optional[Union[Any, Sequence[Any]]] = None,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ):
        return self.defer("reindexFlags", locals())
