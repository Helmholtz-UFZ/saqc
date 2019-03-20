#! /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Any, Optional
from numbers import Number

import numpy as np
import pandas as pd

from lib.types import ArrayLike, T


class AbstractFlagger(ABC):
    def __init__(self, no_flag: T, flag: T):
        self.no_flag: T = no_flag
        self.flag: T = flag

    def setFlag(self,
                flags: ArrayLike,
                flag: Optional[T] = None,
                **kwargs: Any) -> ArrayLike:
        if flag is None:
            flag = self._flag
        flags[:] = flag
        return flags

    def emptyFlags(self,
                   data: pd.DataFrame,
                   value: Optional[Number] = np.nan) -> pd.DataFrame:
        return pd.DataFrame(
            index=data.index,
            columns=data.columns,
            data=value)

    def isFlagged(self, flags: ArrayLike) -> ArrayLike:
        return flags != self.no_flag

    def nextTest(self):
        pass

    def firstTest(self):
        pass
