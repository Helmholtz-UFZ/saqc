#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional
from copy import deepcopy

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
    def __init__(self, categories):
        self.signature = ("flag", "force")
        self.categories = Flags(categories)
        self._flags = None

    # @property
    # def flags(self):
    #     return self._flags

    def initFlags(self, data: pd.DataFrame):
        """
        TODO: rename to initFromData
        """
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        self._flags = self._assureDtype(flags)
        return self

    def initFromFlags(self, flags: pd.DataFrame):
        check_isdf(flags, 'flags', allow_multiindex=False)
        out = deepcopy(self)
        out._flags = out._assureDtype(flags)
        return out

    def setFlagger(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")
        other_flags = other._flags
        out = deepcopy(self)
        # NOTE: I have no idea, why the next statement is failing...
        #       it does however make the loop necessary
        #out._flags.loc[other_flags.index, other_flags.columns] = self._assureDtype(other_flags)
        # TODO: get rid of the loop
        for v in other_flags.columns:
            out._flags.loc[other_flags.index, v] = other_flags[v]
        return out

    def getFlagger(self, loc=None, iloc=None):
        mask = self._locator2Mask(field=slice(None), loc=loc, iloc=iloc)
        out = deepcopy(self)
        out._flags = self._flags[mask]
        return out

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        flags = self.getFlags(field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def getFlags(self, field=None, loc=None, iloc=None, **kwargs):
        # NOTE: needs a copy argument to prevent unnecessary copies
        field = field or slice(None)
        flags = self._flags.copy()
        mask = self._locator2Mask(field, loc, iloc)
        # return flags[field][mask]
        return self._assureDtype(flags.loc[mask, field])

    def _locator2Mask(self, field=None, loc=None, iloc=None):
        locator = [l for l in (loc, iloc, slice(None)) if l is not None][0]
        if np.isscalar(field):
            field = [field]
        flags = self._flags[field]
        mask = pd.Series(
            data=np.zeros(len(flags)),
            dtype=bool,
            index=flags.index,
        )
        mask[locator] = True
        return mask

    def _broadcastFlags(self, field, flag):

        this = self.getFlags(field)

        if np.isscalar(flag):
            flag = np.full(len(this), flag)

        return pd.Series(
            data=flag, index=this.index,
            name=field, dtype=self.categories)

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs):

        flag = self.BAD if flag is None else flag

        this = self.getFlags(field=field)
        other = self._broadcastFlags(field=field, flag=flag)

        mask = self._locator2Mask(field, loc, iloc)
        if not force:
            mask &= (this < other).values

        out = deepcopy(self)
        out._flags.loc[mask, field] = other[mask]
        return out
        # self._flags.loc[mask, field] = other[mask]
        # return self

    def clearFlags(self, field, loc=None, iloc=None, **kwargs):
        if field is None:
            # NOTE: I don't see a need for this restriction
            raise ValueError('field cannot be None')
        return self.setFlags(field=field, loc=loc, iloc=iloc, flag=self.UNFLAGGED, force=True)

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

    def _assureDtype(self, flags):
        if isinstance(flags, pd.Series):
            flags = flags.astype(self.categories)
            return flags

        for c in flags.columns:
            flags[c] = flags[c].astype(self.categories)
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
