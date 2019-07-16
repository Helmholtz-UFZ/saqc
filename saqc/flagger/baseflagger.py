#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..lib.types import PandasLike, ArrayLike, T

COMPARATOR_MAP = {
    "==": op.eq,
    ">=": op.ge,
    ">" : op.gt,
    "<=": op.le,
    "<" : op.lt,
}

class Flags(pd.CategoricalDtype):
    def __init__(self, flags):
        # NOTE: all flag schemes need to support
        #       at least 3 flag categories:
        #       * unflagged
        #       * good
        #       * bad
        assert len(flags) > 2
        super().__init__(flags, ordered=True)

    def unflagged(self):
        return self[0]

    def good(self):
        return self[1]

    def bad(self):
        return self[-1]

    def suspicious(self):
        return self[2:-1]

    def __getitem__(self, idx):
        return self.categories[idx]


class BaseFlagger:
    def __init__(self, flags):
        self.flags = Flags(flags)

    def setFlag(self,
                flags: PandasLike,
                flag: Optional[T] = None,
                **kwargs: Any) -> np.ndarray:

        flag = self.BAD if flag is None else self._checkFlag(flag)
        flags = flags.copy()
        # NOTE:
        # conversion of 'flags' frame to np.array is done here already,
        # since return argument is just the array anyway. For applying
        # mulitdimensional indexing on the DataFrame 'flags', you would
        # have to stack it first
        # (try flags.stack[flags<flag] = flag and than unstack.)
        flags = flags.values
        flags[flags < flag] = flag

        return np.squeeze(flags)

    @property
    def UNFLAGGED(self):
        return self.flags.unflagged()

    @property
    def GOOD(self):
        return self.flags.good()

    @property
    def BAD(self):
        return self.flags.bad()

    @property
    def SUSPICIOUS(self):
        return self.flags.suspicious()

    def initFlags(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            out = pd.Series(data=self.flags[0], index=data.index, name=data.name)
        if isinstance(data, pd.DataFrame):
            out = pd.DataFrame(data=self.flags[0], index=data.index, columns=data.columns)

        # NOTE:
        # astype conversion of return Dataframe performed
        # seperately, because pd.DataFrame(..., dtype=self.flags)
        # wont give you categorical flag objects
        return out.astype(self.flags)

    def isFlagged(self, flags: ArrayLike, flag: T = None, comparator: str = ">") -> ArrayLike:
        cp = COMPARATOR_MAP[comparator]
        if flag is None:
            flag = self.GOOD
        return pd.notnull(flags) & cp(flags, self._checkFlag(flag))

    def _checkFlag(self, flag):
        if flag not in self.flags:
            raise ValueError(
                f"Invalid flag '{flag}'. "
                f"Possible choices are {list(self.flags.categories)[1:]}")
        return flag

    def nextTest(self):
        pass
