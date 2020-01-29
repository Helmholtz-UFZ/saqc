#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import TypeVar, Union, Any

import numpy as np
import pandas as pd

from saqc.lib.tools import toSequence, assertScalar, assertDataFrame


COMPARATOR_MAP = {
    "!=": op.ne,
    "==": op.eq,
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
}


BaseFlaggerT = TypeVar("BaseFlaggerT")
PandasT = Union[pd.Series, pd.DataFrame]
# TODO: get some real types here (could be tricky...)
LocT = Any
IlocT = Any
FlagT = Any


class BaseFlagger(ABC):
    @abstractmethod
    def __init__(self, dtype):
        # NOTE: the type of the _flags DataFrame
        self.dtype = dtype
        # NOTE: the arggumens of setFlags supported from
        #       the configuration functions
        self.signature = ("flag",)
        self._flags: pd.DataFrame

    def initFlags(self, data: pd.DataFrame = None, flags: pd.DataFrame = None) -> BaseFlaggerT:
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        if data is None and flags is None:
            raise TypeError("either 'data' or 'flags' are required")
        if data is not None:
            flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        return self._copy(self._assureDtype(flags))

    def setFlagger(self, other: BaseFlaggerT):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")

        this = self._flags
        other = other._flags

        flags = this.reindex(
            index=this.index.union(other.index),
            columns=this.columns.union(other.columns, sort=False),
            fill_value=self.UNFLAGGED,
        )

        for key, values in other.iteritems():
            flags.loc[other.index, key] = values

        return self._copy(self._assureDtype(flags))

    def getFlagger(self, field: str = None, loc: LocT = None, iloc: IlocT = None) -> BaseFlaggerT:
        """
        return a potentially trimmed down copy of self
        """
        assertScalar("field", field, optional=True)
        mask = self._locatorMask(field=slice(None), loc=loc, iloc=iloc)
        flags = self._flags.loc[mask, field or slice(None)]
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()
        return self._copy(flags)

    def getFlags(self, field: str = None, loc: LocT = None, iloc: IlocT = None) -> PandasT:
        """
        return a copy of a potentially trimmed down 'self._flags' DataFrame
        """
        assertScalar("field", field, optional=True)
        field = field or slice(None)
        flags = self._flags.copy()
        mask = self._locatorMask(field, loc, iloc)
        return flags.loc[mask, field]

    def setFlags(
        self, field: str, loc: LocT = None, iloc: IlocT = None, flag: FlagT = None, force: bool = False, **kwargs,
    ) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)

        flag = self.BAD if flag is None else self._checkFlag(flag)

        this = self.getFlags(field=field)
        other = self._broadcastFlags(field=field, flag=flag)

        mask = self._locatorMask(field, loc, iloc)
        if not force:
            mask &= (this < other).values

        out = deepcopy(self)
        out._flags.loc[mask, field] = other[mask]
        return out

    def clearFlags(self, field: str, loc: LocT = None, iloc: IlocT = None, **kwargs) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)
        return self.setFlags(field=field, loc=loc, iloc=iloc, flag=self.UNFLAGGED, force=True, **kwargs)

    def isFlagged(
        self, field=None, loc: LocT = None, iloc: IlocT = None, flag: FlagT = None, comparator: str = ">", **kwargs,
    ) -> PandasT:
        assertScalar("field", field, optional=True)
        assertScalar("flag", flag, optional=True)
        self._checkFlag(flag)
        flag = self.GOOD if flag is None else flag
        flags = self.getFlags(field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def _copy(self, flags: pd.DataFrame = None) -> BaseFlaggerT:
        out = deepcopy(self)
        if flags is not None:
            out._flags = flags
        return out

    def _locatorMask(self, field: str = None, loc: LocT = None, iloc: IlocT = None) -> PandasT:
        field = field or slice(None)
        locator = [l for l in (loc, iloc, slice(None)) if l is not None][0]
        index = self._flags.index
        mask = pd.Series(data=np.zeros(len(index), dtype=bool), index=index)
        mask[locator] = True
        return mask

    def _broadcastFlags(self, field: str, flag: FlagT) -> pd.Series:

        this = self.getFlags(field)

        if np.isscalar(flag):
            flag = np.full_like(this, flag)

        return pd.Series(data=flag, index=this.index, name=field, dtype=self.dtype)

    def _checkFlag(self, flag):
        if flag is not None and not self._isDtype(flag):
            raise TypeError(f"invalid flag value '{flag}' for flagger 'self.__class__'")
        return flag

    def _assureDtype(self, flags):
        # NOTE: building up new DataFrames is significantly
        #       faster than assigning into existing ones
        if isinstance(flags, pd.Series):
            return flags.astype(self.dtype)
        tmp = OrderedDict()
        for c in flags.columns:
            tmp[c] = flags[c].astype(self.dtype)
        return pd.DataFrame(tmp)

    @abstractmethod
    def _isDtype(self, flag) -> bool:
        pass

    @property
    @abstractmethod
    def UNFLAGGED(self) -> FlagT:
        """ Return the flag that indicates unflagged data """
        pass

    @property
    @abstractmethod
    def GOOD(self) -> FlagT:
        """ Return the flag that indicates the very best data """
        pass

    @property
    @abstractmethod
    def BAD(self) -> FlagT:
        """ Return the flag that indicates the worst data """
        pass

    @abstractmethod
    def isSUSPICIOUS(self, flag: FlagT) -> bool:
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
        pass
