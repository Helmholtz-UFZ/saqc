#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..lib.types import PandasLike, ArrayLike, T
from ..lib.tools import *

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
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.flags[0], index=data.index, columns=data.columns)
        return self._assureDtype(flags)

    def isFlagged(self, flags: PandasLike, flag: T = None, comparator: str = ">") -> PandasLike:
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        check_ispdlike(flags, 'flags', allow_multiindex=False)
        flags = self._assureDtype(flags)
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        cp = COMPARATOR_MAP[comparator]
        isflagged = pd.notna(flags) & cp(flags, flag)
        return isflagged

    def getFlags(self, flags: PandasLike) -> PandasLike:
        check_ispdlike(flags, 'flags', allow_multiindex=False)
        return flags

    def setFlags(self, flags: pd.DataFrame, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        check_isdf(flags, 'flags', allow_multiindex=False)
        # prepare
        flags = self._assureDtype(flags, field).copy()
        flag = self.BAD if flag is None else self._checkFlag(flag)
        flags_loc, rows, col = self._getIndexer(flags, field, loc, iloc)

        # set
        if isinstance(flag, pd.Series):
            if len(flags.index) != len(flags):
                raise ValueError('Length of flags and flag must match')
            i, r, _ = self._getIndexer(flag, field, loc, iloc)
            flag = i[r].squeeze()

        if force:
            mask = [True] * len(rows)
            idx = flags_loc[rows, col].index
        else:
            mask = flags_loc[rows, col] < flag
            idx = mask[mask].index

        if isinstance(flag, pd.Series):
            flag = flag[mask]

        flags.loc[idx, field] = flag
        return self._assureDtype(flags, field)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        check_isdf(flags, 'flags', allow_multiindex=False)
        flags_loc, rows, col = self._getIndexer(flags, field, loc, iloc)
        flags_loc[rows, col] = self.UNFLAGGED
        return self._assureDtype(flags, field)

    def _checkFlag(self, flag):
        if isinstance(flag, pd.Series):
            if not self._isFlagsDtype(flag.dtype):
                raise TypeError(f"flag(-series) is not of expected '{self.flags}'-dtype with ordered categories "
                                f"{list(self.flags.categories)}, '{flag.dtype}'-dtype was passed.")
        else:
            if flag not in self.flags:
                raise ValueError(f"Invalid flag '{flag}'. Possible choices are {list(self.flags.categories)[1:]}")
        return flag

    def _getIndexer(self, flags, field, loc=None, iloc=None):
        if loc is not None and iloc is not None:
            raise ValueError("params `loc` and `iloc` are mutual exclusive")
        elif loc is not None and iloc is None:
            indexer, rows, col = flags.loc, loc, field
        elif loc is None and iloc is not None:
            indexer, rows, col = flags.iloc, iloc, flags.columns.get_loc(field)
        elif loc is None and iloc is None:
            indexer, rows, col = flags.loc, slice(None), field
        return indexer, rows, col

    def _assureDtype(self, flags, field=None):
        if field is None:  # we got a df
            flags = flags.astype(self.flags)
        elif not self._isFlagsDtype(flags[field].dtype):
            flags[field] = flags[field].astype(self.flags)
        return flags

    def _isFlagsDtype(self, dtype):
        return isinstance(dtype, pd.CategoricalDtype) and dtype == self.flags

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
