#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import TypeVar, Union

import numpy as np
import pandas as pd

from saqc.lib.tools import toSequence, assertScalar


COMPARATOR_MAP = {
    "==": op.eq,
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
}


BaseFlaggerT = TypeVar("BaseFlaggerT")
PandasT = Union[pd.Series, pd.DataFrame]


class BaseFlagger(ABC):

    @abstractmethod
    def __init__(self, dtype):
        self.dtype = dtype
        self.signature = ("flag",)
        self._flags: pd.DataFrame

    def initFlags(self, data: pd.DataFrame = None, flags: pd.DataFrame = None) -> BaseFlaggerT:
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """
        pass

    def setFlagger(self, other: BaseFlaggerT):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")
        out = deepcopy(self)
        # NOTE: for a weird reason, this only works with the loop
        for v in other._flags.columns:
            out._flags.loc[other._flags.index, v] = other._flags[v]
        return out

    def getFlagger(self, field: str = None, loc=None, iloc=None) -> BaseFlaggerT:
        """
        return a potentially trimmed down copy of self
        """
        assertScalar("field", field, optional=True)
        mask = self._locatorMask(field=slice(None), loc=loc, iloc=iloc)
        out = deepcopy(self)
        flags = self._flags.loc[mask, field or slice(None)]
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()
        out._flags = flags
        return out

    def getFlags(self, field: str = None, loc=None, iloc=None, **kwargs) -> PandasT:
        """
        return a copy of potentially trimmed down 'self._flags' DataFrame
        """
        assertScalar("field", field, optional=True)
        field = field or slice(None)
        flags = self._flags.copy()
        mask = self._locatorMask(field, loc, iloc)
        return flags.loc[mask, field]

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)

        flag = self.BAD if flag is None else flag

        this = self.getFlags(field=field)
        other = self._broadcastFlags(field=field, flag=flag)

        mask = self._locatorMask(field, loc, iloc)
        if not force:
            mask &= (this < other).values

        out = deepcopy(self)
        out._flags.loc[mask, field] = other[mask]
        return out

    def clearFlags(self, field: str, loc=None, iloc=None, **kwargs) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)
        return self.setFlags(field=field, loc=loc, iloc=iloc, flag=self.UNFLAGGED, force=True)

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs) -> PandasT:
        assertScalar("field", field, optional=True)
        assertScalar("flag", flag, optional=True)
        flag = self.GOOD if flag is None else flag
        flags = self.getFlags(field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def _locatorMask(self, field: str = None, loc=None, iloc=None) -> PandasT:
        field = field or slice(None)
        locator = [l for l in (loc, iloc, slice(None)) if l is not None][0]
        flags = self._flags[toSequence(field)]
        mask = pd.Series(data=np.zeros(len(flags), dtype=bool), index=flags.index)
        mask[locator] = True
        return mask

    def _broadcastFlags(self, field: str, flag) -> PandasT:

        this = self.getFlags(field)

        if np.isscalar(flag):
            flag = np.full_like(this, flag)

        return pd.Series(
            data=flag, index=this.index,
            name=field, dtype=self.dtype)

    def nextTest(self):
        pass

    @property
    @abstractmethod
    def UNFLAGGED(self):
        """ Return the flag that indicates unflagged data """
        pass

    @property
    @abstractmethod
    def GOOD(self):
        """ Return the flag that indicates the very best data """
        pass

    @property
    @abstractmethod
    def BAD(self):
        """ Return the flag that indicates the worst data """
        pass

    @abstractmethod
    def isSUSPICIOUS(self, flag) -> bool:
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
        pass
