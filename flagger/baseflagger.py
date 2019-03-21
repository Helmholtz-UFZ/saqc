#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional
from numbers import Number

import numpy as np
import pandas as pd

from lib.types import PandasLike, ArrayLike, T


class BaseFlagger:
    def __init__(self, no_flag: T, flag: T):
        self.no_flag: T = no_flag
        self.flag: T = flag

    def setFlag(self,
                flags: PandasLike,
                flag: Optional[T] = None,
                **kwargs: Any) -> PandasLike:
        if flag is None:
            flag = self.flag
        flags[:] = flag
        return flags

    def emptyFlags(self,
                   data: pd.DataFrame,
                   value: Optional[Number] = np.nan) -> pd.DataFrame:
        out = data.copy()
        out[:] = value
        return out

    def isFlagged(self, flags: ArrayLike, flag: T = None) -> ArrayLike:
        if flag is None:
            return (pd.notnull(flags) & (flags != self.no_flag))
        return flags == flag

    def nextTest(self):
        pass
