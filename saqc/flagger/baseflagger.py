#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import TypeVar, Union, Any

import numpy as np
import pandas as pd
import dios.dios as dios

from saqc.lib.tools import toSequence, assertScalar, assertDictOfSeries

COMPARATOR_MAP = {
    "!=": op.ne,
    "==": op.eq,
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
}

# TODO: get some real types here (could be tricky...)
LocT = Any
FlagT = Any
diosT = dios.DictOfSeries
BaseFlaggerT = TypeVar("BaseFlaggerT")
PandasT = Union[pd.Series, diosT]


class BaseFlagger(ABC):
    @abstractmethod
    def __init__(self, dtype):
        # NOTE: the type of the _flags DictOfSeries
        self.dtype = dtype
        # NOTE: the arggumens of setFlags supported from
        #       the configuration functions
        self.signature = ("flag",)
        self._flags: diosT = dios.DictOfSeries()

    @property
    def flags(self):
        return self._flags.copy()

    def initFlags(self, data: diosT = None, flags: diosT = None) -> BaseFlaggerT:
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        if data is None and flags is None:
            raise TypeError("either 'data' or 'flags' are required")

        if data is not None:
            assert isinstance(data, diosT)
            flags = data.copy()
            flags[:] = self.UNFLAGGED
        else:
            assert isinstance(flags, diosT)

        newflagger = self.copy()
        newflagger._flags = flags.astype(self.dtype)
        return newflagger

    def setFlagger(self, other: BaseFlaggerT):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")

        this = self.flags
        other = other.flags

        # use dios.merge() as soon as it implemented
        # see https://git.ufz.de/rdm/dios/issues/15
        new = this.copy()
        cols = this.columns.intersection(other.columns)
        for c in cols:
            l, r = this[c], other[c]
            l = l.align(r, join='outer')[0]
            l.loc[r.index] = r
            new[c] = l

        newcols = other.columns.difference(new.columns)
        for c in newcols:
            new[c] = other[c].copy()

        newflagger = self.copy()
        newflagger._flags = new
        return newflagger

    def getFlagger(self, field: str = None, loc: LocT = None) -> BaseFlaggerT:
        """ Return a potentially trimmed down copy of self. """
        flags = self.getFlags(field=field, loc=loc)
        flags = dios.to_dios(flags)
        newflagger = self.copy()
        newflagger._flags = flags
        return newflagger

    def getFlags(self, field: str = None, loc: LocT = None) -> PandasT:
        """ Return a potentially, to `loc`, trimmed down version of flags.

        Return
        ------
        a pd.Series if field is a string or a Dios if not

        Note
        ----
            This is more or less a __getitem__(key)-like function, where
            self._flags is accessed and key is a single key or a tuple.
            Either key is [loc] or [loc,field]. loc also can be a 2D-key,
            aka. a booldios"""

        # loc should be a valid 2D-indexer and
        # then field must be None. Otherwise aloc
        # will fail and throw the correct Error.
        if isinstance(loc, diosT) and field is None:
            indexer = loc

        else:
            loc = slice(None) if loc is None else loc
            field = slice(None) if field is None else self._check_field(field)
            indexer = (loc, field)

        return self.flags.aloc[indexer]

    def setFlags(self, field: str, loc: LocT = None, flag: FlagT = None, force: bool = False, **kwargs) -> BaseFlaggerT:
        """Overwrite existing flags at loc.

        If `force=False` (default) only flags with a lower priority are overwritten,
        otherwise, if `force=True`, flags are overwritten unconditionally.
        """
        assert "iloc" not in kwargs, "deprecated keyword, iloc"

        assertScalar("field", field, optional=False)
        flag = self.BAD if flag is None else flag

        if force:
            row_indexer = loc
        else:
            # trim flags to loc, we always get a pd.Series returned
            this = self.getFlags(field=field, loc=loc)
            row_indexer = this < flag

        out = deepcopy(self)
        out._flags.aloc[row_indexer, field] = flag
        return out

    def clearFlags(self, field: str, loc: LocT = None, **kwargs) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)
        if "force" in kwargs:
            raise ValueError("Keyword 'force' is not allowed here.")
        if "flag" in kwargs:
            raise ValueError("Keyword 'flag' is not allowed here.")
        return self.setFlags(field=field, loc=loc, flag=self.UNFLAGGED, force=True, **kwargs)

    def isFlagged(self, field=None, loc: LocT = None, flag: FlagT = None, comparator: str = ">") -> PandasT:
        assertScalar("flag", flag, optional=True)
        flag = self.GOOD if flag is None else flag
        flags = self.getFlags(field, loc)
        cp = COMPARATOR_MAP[comparator]

        # use notna() to prevent nans to become True,
        # like in: np.nan != 0 -> True
        flagged = flags.notna() & cp(flags, flag)
        return flagged

    def copy(self) -> BaseFlaggerT:
        return deepcopy(self)

    def _check_field(self, field):
        """ Check if (all) field(s) in self._flags. """

        # wait for outcome of
        # https://git.ufz.de/rdm-software/saqc/issues/46
        failed = []
        if isinstance(field, str):
            if field not in self.flags:
                failed += [field]
        else:
            try:
                for f in field:
                    if f not in self.flags:
                        failed += [f]
            # not iterable, probably a slice or
            # any indexer we dont have to check
            except TypeError:
                pass

        if failed:
            raise ValueError(f"key(s) missing in flags: {failed}")
        return field

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
