#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple

import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import Flags
from saqc.core.modules.base import ModuleBase


class Generic(ModuleBase):

    def process(
            self, 
            field: str,
            func: Callable[[pd.Series], pd.Series],
            nodata: float = np.nan,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("process", locals())

    def flag(
            self, 
            field: str,
            func: Callable[[pd.Series], pd.Series],
            nodata: float = np.nan,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flag", locals())
