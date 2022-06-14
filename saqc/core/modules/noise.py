#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.lib.docurator import doc


class Noise:
    @doc(saqc.funcs.noise.flagByStatLowPass.__doc__)
    def flagByStatLowPass(
        self,
        field: str,
        func: Callable[[np.ndarray, pd.Series], float],
        window: str | pd.Timedelta,
        thresh: float,
        sub_window: str | pd.Timedelta = None,
        sub_thresh: float = None,
        min_periods: int = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByStatLowPass", locals())
