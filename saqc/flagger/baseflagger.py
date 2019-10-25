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
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
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

    def initFlags(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be of type pd.DataFrame, {type(data)} was given")
        flags = pd.DataFrame(data=self.flags[0], index=data.index, columns=data.columns)
        flags = flags.astype(self.flags)
        return flags

    def isFlagged(self, flags: PandasLike, flag: T = None, comparator: str = ">") -> PandasLike:
        cp = COMPARATOR_MAP[comparator]
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        isflagged = pd.notna(flags) & cp(flags, flag)
        return isflagged

    def getFlags(self, flags: PandasLike):
        return flags

    def setFlags(self, flags, field, mask_or_indexer=None, flag=None, **kwargs):
        # prepare
        flags = flags.copy()
        r = slice(None) if mask_or_indexer is None else mask_or_indexer
        flag = self.BAD if flag is None else self._checkFlag(flag)
        # set
        mask = flags.loc[r, field] < flag
        idx = mask[mask].index
        flags.loc[idx, field] = flag
        return flags

    def setFlag(self, flags: pd.Series, flag: Optional[T] = None, **kwargs: Any) -> pd.Series:
        flags = self._checkFlagsType(flags)
        flag = self.BAD if flag is None else self._checkFlag(flag)
        flags = flags.copy().values
        # NOTE:
        # - breaks if we loose the pd.Categorical dtype, assert this condition!
        # - there is no way to overwrite with 'better' flags
        mask = flags < flag

        if isinstance(flag, pd.Series):
            flags[mask] = flag[mask]
        else:
            flags[mask] = flag

        return self._finalizeFlags(flags)

    def clearFlags(self, flags: pd.Series, **kwargs) -> pd.Series:
        flags = self._checkFlagsType(flags)
        flags = flags.copy().values
        flags[:] = self.UNFLAGGED
        return self._finalizeFlags(flags)

    def _checkFlag(self, flag):
        if isinstance(flag, pd.Series):
            if flag.dtype != self.flags:
                raise TypeError(f"Passed flags series is of invalid '{flag.dtype}' dtype. "
                                f"Expected {self.flags} type with ordered categories {list(self.flags.categories)}")
        else:
            if flag not in self.flags:
                raise ValueError(f"Invalid flag '{flag}'. Possible choices are {list(self.flags.categories)[1:]}")
        return flag

    def _checkFlagsType(self, flags):
        if isinstance(flags, pd.DataFrame):
            flags = flags.squeeze()
        if not isinstance(flags, pd.Series):
            raise TypeError(f"flags must be of type pd.Series, {type(flags)} was given")
        return flags

    def _finalizeFlags(self, flags: pd.Series):
        if flags.dtype != self.flags:
            nancount = flags.isna().sum()
            flags = flags.astype(self.flags)
            if nancount != flags.isna.sum():
                raise RuntimeError("We lost dtype :(")
        return flags

    def nextTest(self):
        pass

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
