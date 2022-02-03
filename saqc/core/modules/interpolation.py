#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Union, Callable

import numpy as np
import pandas as pd

from saqc.constants import UNFLAGGED
import saqc
from saqc.funcs.interpolation import _SUPPORTED_METHODS
from sphinxdoc.scripts.templates import doc
import saqc.funcs


class Interpolation:
    @doc(saqc.funcs.interpolation.interpolateByRolling.__doc__)
    def interpolateByRolling(
        self,
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("interpolateByRolling", locals())

    @doc(saqc.funcs.interpolation.interpolateInvalid.__doc__)
    def interpolateInvalid(
        self,
        field: str,
        method: _SUPPORTED_METHODS,
        order: int = 2,
        limit: int = 2,
        downgrade: bool = False,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("interpolateInvalid", locals())

    @doc(saqc.funcs.interpolation.interpolateIndex.__doc__)
    def interpolateIndex(
        self,
        field: str,
        freq: str,
        method: _SUPPORTED_METHODS,
        order: int = 2,
        limit: int = 2,
        downgrade: bool = False,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("interpolateIndex", locals())
