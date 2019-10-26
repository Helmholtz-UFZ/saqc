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
        return self._assureDtype(flags)

    def isFlagged(self, flags: PandasLike, flag: T = None, comparator: str = ">") -> PandasLike:
        flags = self._assureDtype(flags)
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        cp = COMPARATOR_MAP[comparator]
        isflagged = pd.notna(flags) & cp(flags, flag)
        return isflagged

    def getFlags(self, flags: PandasLike) -> PandasLike:
        return flags

    def setFlags(self, flags: pd.DataFrame, field, mask_or_indexer=None, flag=None, **kwargs) -> pd.DataFrame:
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        # prepare
        flags = self._assureDtype(flags.copy(), field)
        r = slice(None) if mask_or_indexer is None else mask_or_indexer
        flag = self.BAD if flag is None else self._checkFlag(flag)
        # set
        mask = flags.loc[r, field] < flag
        idx = mask[mask].index
        flags.loc[idx, field] = flag
        return self._assureDtype(flags, field)

    def clearFlags(self, flags, field, mask_or_indexer=None, **kwargs):
        moi = slice(None) if mask_or_indexer is None else mask_or_indexer
        flags.loc[moi, field] = self.UNFLAGGED
        return self._assureDtype(flags, field)

    def _checkFlag(self, flag):
        if isinstance(flag, pd.Series):
            if flag.dtype != self.flags:
                raise TypeError(f"Passed flags series is of invalid '{flag.dtype}' dtype. "
                                f"Expected {self.flags} type with ordered categories {list(self.flags.categories)}")
        else:
            if flag not in self.flags:
                raise ValueError(f"Invalid flag '{flag}'. Possible choices are {list(self.flags.categories)[1:]}")
        return flag

    def _assureDtype(self, flags, field=None):
        if field is None:
            flags = flags.astype(self.flags)
        elif not isinstance(flags[field].dtype, pd.Categorical):
            flags[field] = flags[field].astype(self.flags)
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
