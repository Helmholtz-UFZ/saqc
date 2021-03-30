#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase


class Generic(ModuleBase):

    def process(
            self,
            field: str,
            func: Callable[[pd.Series], pd.Series],
            nodata: float = np.nan,
            **kwargs
    ) -> SaQC:
        return self.defer("process", locals())

    def flag(
            self,
            field: str,
            func: Callable[[pd.Series], pd.Series],
            nodata: float = np.nan,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flag", locals())
