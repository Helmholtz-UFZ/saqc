#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy
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
        stat: Callable[[numpy.array, pd.Series], float],
        winsz: FreqString,
        thresh: PositiveFloat,
        sub_winsz: FreqString = None,
        sub_thresh: PositiveFloat = None,
        min_periods: PositiveInt = None,
        flag: float = BAD,
    ) -> saqc.SaQC:
        return self.defer("flagByStatLowPass", locals())