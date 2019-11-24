#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from collections import OrderedDict
from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar

import numpy as np
import pandas as pd

from saqc.lib.tools import check_isdf

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
        """ Init the class and set the categories (in param flags) that are used here."""
        self.dtype = dtype
        self._flags = None

    def initFlags(self, data: pd.DataFrame):
        """
        TODO: rename to initFromData
        """
        pass

    def initFromFlags(self, flags: pd.DataFrame):
        """
        TODO: merge into initFlags,
              controlled by an optional argument
        """
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

    def getFlagger(self, field=None, loc=None, iloc=None):
        # TODO: multiindex-column flagger implementtions might loose
        #       an index-level here. Take care of that
        mask = self._locatorMask(field=slice(None), loc=loc, iloc=iloc)
        out = deepcopy(self)
        flags = self._flags.loc[mask, field or slice(None)]
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()
        out._flags = flags
        return out

    def getFlags(self, field=None, loc=None, iloc=None, **kwargs):
        # NOTE: needs a copy argument to prevent unnecessary copies
        field = field or slice(None)
        flags = self._flags.copy()
        mask = self._locatorMask(field, loc, iloc)
        return self._assureDtype(flags.loc[mask, field])

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs):

        flag = self.BAD if flag is None else self._checkFlag(flag)

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
            # NOTE: I don't see a need for this restriction
            raise ValueError('field cannot be None')
        return self.setFlags(field=field, loc=loc, iloc=iloc, flag=self.UNFLAGGED, force=True)

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        flags = self.getFlags(field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def _locatorMask(self, field=None, loc=None, iloc=None):
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
            name=field, dtype=self.dtype)

    @abstractmethod
    def _checkFlag(self, flags):
        pass

    @abstractmethod
    def _assureDtype(self, flags):
        pass

    def _isCategorical(self, f) -> bool:
        if isinstance(f, pd.Series):
            return isinstance(f.dtype, pd.CategoricalDtype) and f.dtype == self.dtype
        return f in self.dtype.categories

    def nextTest(self):
        pass

    @abstractproperty
    def UNFLAGGED(self):
        """ Return the flag that indicates unflagged data """
        pass

    @abstractproperty
    def GOOD(self):
        """ Return the flag that indicates the very best data """
        pass

    @abstractproperty
    def BAD(self):
        """ Return the flag that indicates the worst data """
        pass

    @abstractmethod
    def isSUSPICIOUS(self, flag):
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
        pass
