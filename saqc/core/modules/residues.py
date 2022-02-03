#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import BAD
import saqc
from sphinxdoc.scripts.templates import doc
import saqc.funcs


class Residues:
    @doc(saqc.funcs.residues.calculatePolynomialResidues.__doc__)
    def calculatePolynomialResidues(
        self,
        field: str,
        window: Union[str, int],
        order: int,
        min_periods: Optional[int] = 0,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("calculatePolynomialResidues", locals())

    @doc(saqc.funcs.residues.calculateRollingResidues.__doc__)
    def calculateRollingResidues(
        self,
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: Optional[int] = 0,
        center: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("calculateRollingResidues", locals())
