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
    def __init__(self, flag_categories):
        self.categories = Flags(flag_categories)
        self._flags = None
        # todo move initFlags code here

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.categories[0], index=data.index, columns=data.columns)
        self._flags = self._assureDtype(flags)

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        flags = self.getFlags(field, loc, iloc)
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        cp = COMPARATOR_MAP[comparator]
        isflagged = pd.notna(flags) & cp(flags, flag)
        return isflagged

    def getFlags(self, field=None, loc=None, iloc=None, **kwargs):
        flags = self._flags if field is None else self._flags[self._checkField(field)]
        locator, rows, _ = self._getLocator(flags, field, loc, iloc)
        flags = locator[rows]
        return self._assureDtype(flags, field)

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        # prepare
        self._checkField(field)
        src = self.BAD if flag is None else self._checkFlag(flag)
        dest = self._flags
        if not isinstance(src, pd.Series):
            src = pd.Series(data=src, index=dest.index)

        # get locations on src
        i, r, _ = self._getLocator(src, field, loc, iloc)
        src = i[r].squeeze()

        # get locations on dest
        dest_loc, rows, col = self._getLocator(self._flags, field, loc, iloc)
        if force:
            idx = dest_loc[rows, col].index
        else:
            mask = dest_loc[rows, col] < src
            idx = mask[mask].index
            # do also shrink src, to fit dest !
            src = src[mask]

        # actually set src to dest
        dest.loc[idx, field] = src
        self._flags = self._assureDtype(dest, field)

    def clearFlags(self, field, loc=None, iloc=None, **kwargs):
        self._checkField(field)
        flags_loc, rows, col = self._getLocator(self._flags, field, loc, iloc)
        flags_loc[rows, col] = self.UNFLAGGED

    def _checkField(self, field):
        if field not in self._flags:
            raise KeyError(f"field {field} is not in flags")
        return field

    def _checkFlag(self, flag, allow_series=False):
        if isinstance(flag, pd.Series):
            if not allow_series:
                raise TypeError('series of flags are not allowed here')

            if len(flag.index) != len(self._flags.index):
                raise ValueError(f'Length of flags ({len(self._flags.index)}) and flag ({len(flag.index)}) must match')

            if not self._isFlagsDtype(flag):
                raise TypeError(f"flag(-series) is not of expected '{self.categories}'-dtype with ordered categories "
                                f"{list(self.categories.categories)}, '{flag.dtype}'-dtype was passed.")
        else:
            if flag not in self.categories:
                raise ValueError(f"Invalid flag '{flag}'. Possible choices are {list(self.categories.categories)}")
        return flag

    def _getLocator(self, flags, field, loc=None, iloc=None):
        if loc is not None and iloc is not None:
            raise ValueError("params `loc` and `iloc` are mutual exclusive")
        elif loc is not None and iloc is None:
            indexer, rows, col = flags.loc, loc, field or slice(None)
        elif loc is None and iloc is not None:
            field = slice(None) if field is None else flags.columns.get_loc(field)
            indexer, rows, col = flags.iloc, iloc, field
        elif loc is None and iloc is None:
            indexer, rows, col = flags.loc, slice(None), field or slice(None)
        return indexer, rows, col

    def _assureDtype(self, flags, field=None):
        if isinstance(flags, pd.Series):
            return flags if self._isFlagsDtype(flags) else flags.astype(self.categories)

        else:  # got a df, recurse
            if field is None:
                for c in flags:
                    flags[c] = self._assureDtype(flags[c])
            else:
                flags[field] = self._assureDtype(flags[field])
        return flags

    def _isFlagsDtype(self, series):
        return isinstance(series.dtype, pd.CategoricalDtype) and series.dtype == self.categories

    def nextTest(self):
        pass

    @property
    def UNFLAGGED(self):
        return self.categories.unflagged()

    @property
    def GOOD(self):
        return self.categories.good()

    @property
    def BAD(self):
        return self.categories.bad()

    @property
    def SUSPICIOUS(self):
        return self.categories.suspicious()
