#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar

import numpy as np
import pandas as pd

from saqc.lib.tools import toSequence

newT = TypeVar("newT")

COMPARATOR_MAP = {
    "==": op.eq,
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
}


class BaseFlagger(ABC):

    @abstractmethod
    def __init__(self, dtype):
        self.dtype = dtype
        self._flags: pd.DataFrame

    def initFlags(self, data: pd.DataFrame = None, flags: pd.DataFrame = None):
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """
        pass

    def setFlagger(self, other):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")
        out = deepcopy(self)
        # NOTE: I have no idea, why the next statement is failing...
        #       it does however make the loop necessary
        #out._flags.loc[other_flags.index, other_flags.columns] = other_flags
        for v in other._flags.columns:
            out._flags.loc[other._flags.index, v] = other._flags[v]
        return out

    def getFlagger(self, field=None, loc=None, iloc=None):
        """
        return a potentially trimmed down copy of self
        """
        mask = self._locatorMask(field=slice(None), loc=loc, iloc=iloc)
        out = deepcopy(self)
        flags = self._flags.loc[mask, field or slice(None)]
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()
        out._flags = flags
        return out

    def getFlags(self, field=None, loc=None, iloc=None, **kwargs):
        """
        return a copy of potentially trimmed down 'self._flags' DataFrame

        NOTE
        Maybe we should add a 'copy' argument to prevent unnecessary copies
        of `self._flags`. As there are now performance issues with the
        current implementation, that's probably not that important or even
        desirable?
        """
        field = field or slice(None)
        flags = self._flags.copy()
        mask = self._locatorMask(field, loc, iloc)
        return flags.loc[mask, field]

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')

        flag = self.BAD if flag is None else flag

        this = self.getFlags(field=field)
        other = self._broadcastFlags(field=field, flag=flag)

        mask = self._locatorMask(field, loc, iloc)
        if not force:
            mask &= (this < other).values

        out = deepcopy(self)
        out._flags.loc[mask, field] = other[mask]
        return out

    def clearFlags(self, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        return self.setFlags(field=field, loc=loc, iloc=iloc, flag=self.UNFLAGGED, force=True)

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        flag = self.GOOD if flag is None else flag
        flags = self.getFlags(field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def _locatorMask(self, field, loc=None, iloc=None):
        # NOTE: get the first non-None locator option or a default slice(None)
        locator = [l for l in (loc, iloc, slice(None)) if l is not None][0]
        flags = self._flags[toSequence(field)]
        mask = pd.Series(data=np.zeros(len(flags), dtype=bool), index=flags.index)
        mask[locator] = True
        return mask

    def _broadcastFlags(self, field, flag):

        this = self.getFlags(field)

        if np.isscalar(flag):
            flag = np.full(len(this), flag)

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
    def isSUSPICIOUS(self, flag):
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
        pass
