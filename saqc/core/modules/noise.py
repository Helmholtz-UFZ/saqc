#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.lib.types import FreqString


class Noise(ModuleBase):
    def flagByStatLowPass(
        self,
        field: str,
        func: Callable[[np.ndarray, pd.Series], float],
        window: FreqString,
        thresh: float,
        sub_window: FreqString = None,
        sub_thresh: float = None,
        min_periods: int = None,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagByStatLowPass", locals())
