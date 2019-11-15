#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional

import numpy as np
import pandas as pd

from .template import FlaggerTemplate
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


class BaseFlagger(FlaggerTemplate):
    def __init__(self, flags):
        self.signature = ("flag", "force")
        self.categories = Flags(flags)

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.categories[0], index=data.index, columns=data.columns)
        return self._assureDtype(flags)

    def isFlagged(self, flags, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        flags = self.getFlags(flags, field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def getFlags(self, flags, field=None, loc=None, iloc=None, **kwargs):
        flags = flags.copy()
        flags = self._checkFlags(flags, **kwargs)
        flags = self._reduceColumns(flags, **kwargs)
        flags = self._reduceRows(flags, field, loc, iloc, **kwargs)
        flags = self._assureDtype(flags, field, **kwargs)
        return flags

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        # in: df, out: df, can modify just one (!) (flag-)column
        if field is None:
            raise ValueError('field cannot be None')
        flags = flags.copy()
        dest = self._checkFlags(flags, **kwargs)
        dest = self._reduceColumns(dest, **kwargs)
        dest_len = len(dest)
        dest = self._reduceRows(dest, field, loc, iloc, **kwargs)
        dest = self._assureDtype(dest, field, **kwargs)
        assert isinstance(dest, pd.Series)

        # prepare src
        src = self.BAD if flag is None else self._checkFlag(flag, allow_series=True, lenght=dest_len)
        if not isinstance(src, pd.Series):
            src = pd.Series(data=src, index=dest.index)
        src = self._reduceRows(src, None, loc, iloc, **kwargs)
        src = self._assureDtype(src, **kwargs)

        # now src and dest are equal-length pd.Series with correct categorical dtype
        # assert isinstance(dest, pd.Series) and isinstance(src, pd.Series) and len(src) == len(dest)
        if force:
            idx = dest.index
        else:
            mask = dest < src
            idx = dest[mask].index
            src = src[mask]

        flags = self._writeFlags(flags, idx, field, src, **kwargs)
        return self._assureDtype(flags, field, **kwargs)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        flags = flags.copy()
        f = self.getFlags(flags, field, loc, iloc, **kwargs)
        flags = self._writeFlags(flags, f.index, field, flag=self.UNFLAGGED, **kwargs)
        return self._assureDtype(flags)

    def _reduceColumns(self, df, field=None, **kwargs) -> pd.DataFrame:
        # in: ?, out: df
        return df

    def _reduceRows(self, df_or_ser, field, loc, iloc, **kwargs) -> pd.DataFrame:
        # in: df, out: df(w/. field), ser(w/o. field), reduced in rows, mismatched loc/iloc: empty df/ser
        df_or_ser = df_or_ser if field is None else df_or_ser[field]
        if loc is not None and iloc is not None:
            raise ValueError("params `loc` and `iloc` are mutual exclusive")
        elif loc is not None and iloc is None:
            return df_or_ser.loc[loc]
        elif loc is None and iloc is not None:
            return df_or_ser.iloc[iloc]
        elif loc is None and iloc is None:
            return df_or_ser

    def _writeFlags(self, flags, rowindex, field, flag, **kwargs):
        # in: df, out: df, w/ modified values
        flags.loc[rowindex, field] = flag
        return flags

    def _checkFlags(self, flags, **kwargs):
        check_isdf(flags, argname='flags')
        return flags

    def _checkFlag(self, flag, allow_series=False, lenght=None):
        if flag is None:
            raise ValueError("flag cannot be None")

        if isinstance(flag, pd.Series):
            if not allow_series:
                raise TypeError('series of flags are not allowed here')

            if not self._isSelfCategoricalType(flag):
                raise TypeError(f"flag(-series) is not of expected '{self.categories}'-dtype with ordered categories "
                                f"{list(self.categories.categories)}, '{flag.dtype}'-dtype was passed.")

            assert lenght is not None, 'faulty Implementation, length param must be given if flag is a series'
            if len(flag) != lenght:
                raise ValueError(f'length of flags ({lenght}) and flag ({len(flag)}) must match, if flag is '
                                 f'a series')

        elif not self._isSelfCategoricalType(flag):
            raise TypeError(f"Invalid flag '{flag}'. Possible choices are {list(self.categories.categories)}")

        return flag

    def _assureDtype(self, flags, field=None, **kwargs):
        # in: df/ser, out: df/ser, affect only the minimal set of columns
        if isinstance(flags, pd.Series):
            return flags if self._isSelfCategoricalType(flags) else flags.astype(self.categories)

        elif isinstance(flags, pd.DataFrame):
            if field is None:
                for c in flags:
                    flags[c] = self._assureDtype(flags[c], **kwargs)
            else:
                flags[field] = self._assureDtype(flags[field], **kwargs)
        else:
            raise NotImplementedError

        return flags

    def _isSelfCategoricalType(self, f) -> bool:
        if isinstance(f, pd.Series):
            return isinstance(f.dtype, pd.CategoricalDtype) and f.dtype == self.categories
        else:
            return f in self.categories

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
