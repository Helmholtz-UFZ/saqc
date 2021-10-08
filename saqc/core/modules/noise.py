#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.lib.types import FreqString, IntegerWindow, ColumnName
from saqc.lib.types import ColumnName, FreqString, PositiveInt, PositiveFloat


class Noise(ModuleBase):
    def flagByStatLowPass(
        self,
        field: ColumnName,
        func: Callable[[np.array, pd.Series], float],
        window: FreqString,
        thresh: PositiveFloat,
        sub_window: FreqString = None,
        sub_thresh: PositiveFloat = None,
        min_periods: PositiveInt = None,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagByStatLowPass", locals())
