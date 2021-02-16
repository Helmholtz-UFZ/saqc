#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np
import pandas as pd

from saqc.core.modules.base import ModuleBase


class Generic(ModuleBase):

    def process(self, field: str, func: Callable[[pd.Series], pd.Series], nodata: float = np.nan, **kwargs):
        return self.defer("process", locals())

    def flag(self, field: str, func: Callable[[pd.Series], pd.Series], nodata: float = np.nan, **kwargs):
        return self.defer("flag", locals())
