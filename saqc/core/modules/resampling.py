#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.funcs.interpolation import _SUPPORTED_METHODS
from saqc.lib.docurator import doc


class Resampling:
    @doc(saqc.funcs.resampling.linear.__doc__)
    def linear(
        self,
        field: str,
        freq: str,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("linear", locals())

    @doc(saqc.funcs.resampling.interpolate.__doc__)
    def interpolate(
        self,
        field: str,
        freq: str,
        method: _SUPPORTED_METHODS,
        order: int = 1,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("interpolate", locals())

    @doc(saqc.funcs.resampling.shift.__doc__)
    def shift(
        self,
        field: str,
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("shift", locals())

    @doc(saqc.funcs.resampling.resample.__doc__)
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

    @doc(saqc.funcs.resampling.concatFlags.__doc__)
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
            "match",
        ] = "match",
        freq: Optional[str] = None,
        drop: Optional[bool] = False,
        squeeze: Optional[bool] = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("concatFlags", locals())
