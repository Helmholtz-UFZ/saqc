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
        # todo move initFlags code here

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.categories[0], index=data.index, columns=data.columns)
        return self._assureDtype(flags)

    def isFlagged(self, flags, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        """
        Return bool information on flags.

        :param flags: pd.Dataframe only
        :param field: None or str. Labelbased column indexer.
        :return: pd.Dataframe if field is None, pd.Series otherwise
        """
        flags = self.getFlags(flags, field, loc, iloc)
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        cp = COMPARATOR_MAP[comparator]
        isflagged = pd.notna(flags) & cp(flags, flag)
        return isflagged

    def getFlags(self, flags, field=None, loc=None, iloc=None, **kwargs):
        """
        Return flags information.

        :param flags: pd.Dataframe only
        :param field: None or str. Labelbased column indexer.
        :param loc: mask or bool-array or Series used as row indexer (see. [1]). Mutual exclusive with `iloc`
        :param iloc: mask or bool-array or int-array used as relative row indexer (see. [2]).
            Mutual exclusive with `loc`
        :param kwargs: unused

        :return: pd.Dataframe if field is None, pd.Series otherwise

        Note: [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html

        Note: [2] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
        """
        self._checkFlags(flags)
        flags = flags if field is None else flags[field]
        locator, rows = self._getRowIndexer(flags, loc, iloc)
        flags = locator[rows]
        return self._assureDtype(flags, field)

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        dest_len = len(flags.index)
        dest = self.getFlags(flags, field, loc, iloc, **kwargs)

        # prepare src
        src = self.BAD if flag is None else self._checkFlag(flag, allow_series=True)
        if isinstance(src, pd.Series):
            if len(src.index) != dest_len:
                raise ValueError(f'Length of flags ({dest_len}) and flag ({len(flag.index)}) must match, if flag is '
                                 f'not a scalar')
        else:
            src = pd.Series(data=src, index=dest.index)
            src = self._assureDtype(src)
        i, r = self._getRowIndexer(src, loc, iloc)
        src = i[r].squeeze()

        if force:
            idx = dest.index
        else:
            mask = dest < src
            idx = dest[mask].index
            src = src[mask]

        flags = self._setFlags(flags, idx, field, src, **kwargs)
        return self._assureDtype(flags, field)

    def _setFlags(self, flags, rowindex, field, values, **kwargs):
        flags.loc[rowindex, field] = values
        return flags

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        f = self.getFlags(flags, field, loc, iloc, **kwargs)
        vals = self.UNFLAGGED
        self._setFlags(flags, f.index, field, vals, **kwargs)

    def _checkFlags(self, flags, **kwargs):
        check_isdf(flags, argname='flags')
        return flags

    def _checkFlag(self, flag, allow_series=False):
        if isinstance(flag, pd.Series):
            if not allow_series:
                raise TypeError('series of flags are not allowed here')

            if not self._isFlagsDtype(flag):
                raise TypeError(f"flag(-series) is not of expected '{self.categories}'-dtype with ordered categories "
                                f"{list(self.categories.categories)}, '{flag.dtype}'-dtype was passed.")
        else:
            if flag not in self.categories:
                if flag is None:
                    raise KeyError("flag cannot be None")
                raise ValueError(f"Invalid flag '{flag}'. Possible choices are {list(self.categories.categories)}")
        return flag

    def _getRowIndexer(self, flags, loc=None, iloc=None):
        if loc is not None and iloc is not None:
            raise ValueError("params `loc` and `iloc` are mutual exclusive")
        elif loc is not None and iloc is None:
            indexer, rows = flags.loc, loc
        elif loc is None and iloc is not None:
            indexer, rows = flags.iloc, iloc
        elif loc is None and iloc is None:
            indexer, rows = flags.loc, slice(None)
        return indexer, rows

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
