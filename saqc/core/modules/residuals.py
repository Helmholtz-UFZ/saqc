#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.lib.docurator import doc


class Residuals:
    @doc(saqc.funcs.residuals.calculatePolynomialResiduals.__doc__)
    def calculatePolynomialResiduals(
        self,
        field: str,
        window: Union[str, int],
        order: int,
        min_periods: Optional[int] = 0,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("calculatePolynomialResiduals", locals())

    @doc(saqc.funcs.residuals.calculateRollingResiduals.__doc__)
    def calculateRollingResiduals(
        self,
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: Optional[int] = 0,
        center: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("calculateRollingResiduals", locals())
