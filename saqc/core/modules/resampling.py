#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
import saqc
from saqc.funcs.interpolation import _SUPPORTED_METHODS


class Resampling:
    def linear(
        self,
        field: str,
        freq: str,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("linear", locals())

    def interpolate(
        self,
        field: str,
        freq: str,
        method: _SUPPORTED_METHODS,
        order: int = 1,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("interpolate", locals())

    def shift(
        self,
        field: str,
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("shift", locals())

    def resample(
        self,
        field: str,
        freq: str,
        func: Callable[[pd.Series], pd.Series] = np.mean,
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        maxna: Optional[int] = None,
        maxna_group: Optional[int] = None,
        maxna_flags: Optional[int] = None,  # TODO: still a case ??
        maxna_group_flags: Optional[int] = None,
        flag_func: Callable[[pd.Series], float] = max,
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("resample", locals())

    def concatFlags(
        self,
        field: str,
        target: str,
        method: Literal[
            "inverse_fagg",
            "inverse_bagg",
            "inverse_nagg",
            "inverse_fshift",
            "inverse_bshift",
            "inverse_nshift",
            "inverse_interpolation",
        ],
        freq: Optional[str] = None,
        drop: Optional[bool] = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("concatFlags", locals())
