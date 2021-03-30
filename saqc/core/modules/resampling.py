#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
from saqc.funcs.interpolation import _SUPPORTED_METHODS


class Resampling(ModuleBase):

    def aggregate(
            self, 
            field: str,
            freq: str,
            value_func,
            flag_func: Callable[[pd.Series], float] = np.nanmax,
            method: Literal["fagg", "bagg", "nagg"] = "nagg",
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("aggregate", locals())

    def linear(
            self, 
            field: str,
            freq: str,
            **kwargs
    ) -> SaQC:
        return self.defer("linear", locals())

    def interpolate(
            self, 
            field: str,
            freq: str,
            method: _SUPPORTED_METHODS,
            order: int = 1,
            **kwargs,
    ) -> SaQC:
        return self.defer("interpolate", locals())

    def mapToOriginal(
            self, 
            field: str,
            method: Literal[
                "inverse_fagg", "inverse_bagg", "inverse_nagg",
                "inverse_fshift", "inverse_bshift", "inverse_nshift",
                "inverse_interpolation"
            ],
            **kwargs
    ) -> SaQC:
        return self.defer("mapToOriginal", locals())

    def shift(
            self, 
            field: str,
            freq: str,
            method: Literal["fshift", "bshift", "nshift"] = "nshift",
            freq_check: Optional[Literal["check", "auto"]] = None,  # TODO: not a user decision
            **kwargs
    ) -> SaQC:
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
            flag_agg_func: Callable[[pd.Series], float] = max,
            freq_check: Optional[Literal["check", "auto"]] = None,
            **kwargs
    ) -> SaQC:
        return self.defer("resample", locals())

    def reindexFlags(
            self, 
            field: str,
            method: Literal[
                "inverse_fagg", "inverse_bagg", "inverse_nagg",
                "inverse_fshift", "inverse_bshift", "inverse_nshift"
            ],
            source: str,
            freq: Optional[str] = None,
            **kwargs
    ) -> SaQC:
        return self.defer("reindexFlags", locals())
