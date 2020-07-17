#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from abc import ABC, abstractmethod

from typing import TypeVar, Union, Any, List, Optional

import pandas as pd
import dios.dios as dios

from saqc.lib.tools import assertScalar, mergeDios, toSequence, mutateIndex

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
FieldsT = Union[str, List[str]]


class BaseFlagger(ABC):
    @abstractmethod
    def __init__(self, dtype):
        # NOTE: the type of the _flags DictOfSeries
        self.dtype = dtype
        # NOTE: the arggumens of setFlags supported from
        #       the configuration functions
        self.signature = ("flag",)
        self._flags: Optional[diosT] = None

    @property
    def initialized(self):
        return self._flags is not None

    @property
    def flags(self):
        return self._flags.copy()

    def initFlags(self, data: diosT = None, flags: diosT = None) -> BaseFlaggerT:
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFLAGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        if data is None and flags is None:
            raise TypeError("either 'data' or 'flags' are required")

        if data is not None and flags is not None:
            raise TypeError("either 'data' or 'flags' can be given")

        if data is not None:
            if not isinstance(data, diosT):
                data = dios.DictOfSeries(data)
            flags = data.copy()
            flags[:] = self.UNFLAGGED
        else:
            if not isinstance(data, diosT):
                flags = dios.DictOfSeries(flags)

        newflagger = self.copy()
        newflagger._flags = flags.astype(self.dtype)
        return newflagger

    def rename(self, field: str, new_name: str):
        newflagger = self.copy()
        newflagger._flags.columns = mutateIndex(newflagger._flags.columns, field, new_name)
        return newflagger

    def merge(self, other: BaseFlaggerT, join: str = "merge"):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")

        newflagger = self.copy(flags=mergeDios(self.flags, other.flags, join=join))
        return newflagger

    def slice(self, field: FieldsT = None, loc: LocT = None, drop: FieldsT = None) -> BaseFlaggerT:
        """ Return a potentially trimmed down copy of self. """
        if drop is not None:
            if field is not None:
                raise TypeError("either 'field' or 'drop' can be given, but not both")
            field = self._flags.columns.drop(drop, errors="ignore")
        flags = self.getFlags(field=field, loc=loc)
        flags = dios.to_dios(flags)
        newflagger = self.copy(flags=flags)
        return newflagger

    def getFlags(self, field: FieldsT = None, loc: LocT = None) -> PandasT:
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
        assert "iloc" not in kwargs, "deprecated keyword, `iloc=slice(i:j)`. Use eg. `loc=srs.index[i:j]` instead."

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
        """
        Returns boolean data that indicate where data has been flagged.

        Parameters
        ----------
        field : str, list-like, default None
            The field(s)/column(s) of the data to be tested if flagged.
            If None all columns are used.

        loc : mask, slice, pd.Index, etc., default None
            The location/rows of the data to be tested if flagged.
            If None all rows are used.

        flag : str, category, list-like, default None
            The flag(s) that define data as flagged. If None, `flagger.GOOD`
            is used.

        comparator : {'<', '<=', '==', '!=', '>=', '>'}, default '>'
            Defines how the comparison is done. The `flags` are always on the
            left-hand-side, thus, the flag to compare is always on the right-
            hand-side. Eg. a call with all defaults, return the equivalent
            of `flagger.getFlags() > flagger.GOOD`

        Returns
        -------
        pandas.Series or dios.DictOfSeries : Return Series if field is a scalar,
        otherwise DictOfSeries.
        """
        if isinstance(flag, pd.Series):
            raise TypeError("flag: pd.Series is not allowed")
        flags_to_compare = set(toSequence(flag, self.GOOD))

        flags = self.getFlags(field, loc)
        cp = COMPARATOR_MAP[comparator]

        # notna() to prevent nans to become True, eg.: `np.nan != 0 -> True`
        flagged = flags.notna()
        for f in flags_to_compare:
            if not self.isValidFlag(f):
                raise ValueError(f"invalid flag: {f}")
            flagged &= cp(flags, f)

        return flagged

    def copy(self, flags=None) -> BaseFlaggerT:
        out = deepcopy(self)
        if flags is not None:
            out._flags = flags
        return out

    def isValidFlag(self, flag: FlagT) -> bool:
        """
        Check if given flag is known to this flagger.

        Parameters
        ----------
        flag: str
            The flag to be checked.

        Returns
        -------
        bool
        """
        # This is a very rudimentary fallback for the check
        # and the child flagger may should implement a better
        # version of it.
        return flag == self.BAD or flag == self.GOOD or flag == self.UNFLAGGED or self.isSUSPICIOUS(flag)

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

    @property
    @abstractmethod
    def GOOD(self) -> FlagT:
        """ Return the flag that indicates the very best data """

    @property
    @abstractmethod
    def BAD(self) -> FlagT:
        """ Return the flag that indicates the worst data """

    @abstractmethod
    def isSUSPICIOUS(self, flag: FlagT) -> bool:
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
