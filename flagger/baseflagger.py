#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.types import PandasLike, ArrayLike, T


class Flags(pd.CategoricalDtype):
    def __init__(self, flags):
        assert len(flags) > 2
        super().__init__(flags, ordered=True)

    def min(self):
        return self[2]

    def max(self):
        return self[-1]

    def __getitem__(self, idx):
        return self.categories[idx]


class BaseFlagger:
    def __init__(self, flags):
        self.flags = Flags(flags)

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
            flag = self.flags[-1]
        flags[flags < flag] = flag
        return flags.values

    def initFlags(self, data: pd.DataFrame) -> pd.DataFrame:
        # out = data.copy() # .astype(self)
        out = data.copy().astype(self.flags)
        out.loc[:] = self.flags[0]
        return out

    def isFlagged(self, flags: ArrayLike, flag: T = None) -> ArrayLike:
        if flag is None:
            return (pd.notnull(flags) & (flags > self.flags[1]))
        return flags >= flag

    def nextTest(self):
        pass
