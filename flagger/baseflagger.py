#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Number
from typing import Any, Optional

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
                **kwargs: Any) -> np.ndarray:
        """
        NOTE:
        this method should really return a numpy array, as
        pandas data structures tend to behave unpredictively
        in assignments, especially if a multi column index is used
        """
        if flag is None:
            flag = self.flag
        flags[:] = flag
        return flags.values

    def initFlags(self,
                  data: pd.DataFrame,
                  value: Optional[Number] = np.nan) -> pd.DataFrame:
        out = data.copy()
        out[:] = value
        return out

    def emptyFlags(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)

    def isFlagged(self, flags: ArrayLike, flag: T = None) -> ArrayLike:
        if flag is None:
            return (pd.notnull(flags) & (flags != self.no_flag))
        return flags == flag

    def nextTest(self):
        pass
