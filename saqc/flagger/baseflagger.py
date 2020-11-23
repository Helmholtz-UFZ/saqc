#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from copy import deepcopy
from abc import ABC, abstractmethod

from typing import TypeVar, Union, Any, List, Optional

import pandas as pd
import numpy as np
import dios

from saqc.lib.tools import assertScalar, mergeDios, toSequence, mutateIndex, customRolling

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
        self.extra_defaults = dict()
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

            flags = dios.DictOfSeries(columns=data.columns)
            for c in flags.columns:
                flags[c] = pd.Series(self.UNFLAGGED, index=data[c].index)
        else:
            if not isinstance(flags, diosT):
                flags = dios.DictOfSeries(flags)

        flags = flags.astype(self.dtype)
        newflagger = self.copy(flags=flags)
        return newflagger

    def merge(self, other: BaseFlaggerT, subset: Optional[List] = None, join: str = "merge", inplace=False):
        """
        Merge the given flagger 'other' into self
        """
        # NOTE: add more checks !?
        if not isinstance(other, self.__class__):
            raise TypeError(f"flagger of type '{self.__class__}' needed")

        if inplace:
            self._flags = mergeDios(self._flags, other._flags, subset=subset, join=join)
            return self
        else:
            return self.copy(flags=mergeDios(self._flags, other._flags, subset=subset, join=join))

    def slice(self, field: FieldsT = None, loc: LocT = None, drop: FieldsT = None, inplace=False) -> BaseFlaggerT:
        """ Return a potentially trimmed down copy of self. """
        if drop is not None:
            if field is not None:
                raise TypeError("either 'field' or 'drop' can be given, but not both")
            field = self._flags.columns.drop(drop, errors="ignore")
        flags = self.getFlags(field=field, loc=loc).to_dios()

        if inplace:
            self._flags = flags
            return self
        else:
            return self.copy(flags=flags)

    def getFlags(self, field: FieldsT = None, loc: LocT = None, full=False):
        """ Return a potentially, to `loc`, trimmed down version of flags.

        Parameters
        ----------
        field : str, list of str or None, default None
            Field(s) to request.
        loc :
            limit result to specific rows.
        full : object
            If True, an additional dict is returned, holding all extras that
            the flagger may specify. These extras can be feed back to a/the
            flagger with `setFlags(...with_extras=True)`.

        Return
        ------
        flags: pandas.Series or dios.DictOfSeries
            If field is a scalar a series is returned, otherwise a dios.
        extras: dict
            Present only if `full=True`. A dict that hold all extra information.

        Note
        ----
        This is more or less a __getitem__(key)-like function, where
        self._flags is accessed and key is a single key or a tuple.
        Either key is [loc] or [loc,field]. loc also can be a 2D-key,
        aka. a booldios

        The resulting dict (full=True) can be feed to setFlags to update extra Columns.
        but field must be a scalar then, because setFlags only can process a scalar field.
        """

        # loc should be a valid 2D-indexer and
        # then field must be None. Otherwise aloc
        # will fail and throw the correct Error.
        if isinstance(loc, diosT) and field is None:
            indexer = loc

        else:
            loc = slice(None) if loc is None else loc
            field = slice(None) if field is None else self._check_field(field)
            indexer = (loc, field)

        # this is a bug in `dios.aloc`, which may return a shallow copied dios, if `slice(None)` is passed
        # as row indexer. Thus is because pandas `.loc` return a shallow copy if a null-slice is passed to a series.
        flags = self._flags.aloc[indexer].copy()
        if full:
            return flags, dict()
        else:
            return flags

    def setFlags(
            self,
            field: str,
            loc: LocT = None,
            flag: FlagT = None,
            force: bool = False,
            inplace: bool = False,
            with_extra: bool = False,
            flag_after: Union[str, int] = None,
            flag_before: Union[str, int] = None,
            win_flag: FlagT = None,
            **kwargs
    ) -> BaseFlaggerT:
        """Overwrite existing flags at loc.

        If `force=False` (default) only flags with a lower priority are overwritten,
        otherwise, if `force=True`, flags are overwritten unconditionally.

        Examples
        --------
        One can use this to update extra columns without knowing their names. Eg. like so:

        >>> field = 'var0'
        >>> flags, extra = flagger.getFlags(field, full=True)
        >>> newflags = magic_that_alter_index(flags)
        >>> for k, v in extra.items()
        ...     extra[k] = magic_that_alter_index(v)
        >>> flagger = flagger.setFlags(field, flags=newflags, with_extra=True, **extra)
        """

        assert "iloc" not in kwargs, "deprecated keyword, `iloc=slice(i:j)`. Use eg. `loc=srs.index[i:j]` instead."

        assertScalar("field", self._check_field(field), optional=False)
        flag = self.BAD if flag is None else flag
        out = self if inplace else deepcopy(self)

        if with_extra and not isinstance(flag, pd.Series):
            raise ValueError("flags must be pd.Series if `with_extras=True`.")

        trimmed = self.getFlags(field=field, loc=loc)
        if force:
            mask = pd.Series(True, index=trimmed.index, dtype=bool)
        else:
            mask = trimmed < flag

        # set flags of the test
        out._flags.aloc[mask, field] = flag

        # calc and set window flags
        if flag_after is not None or flag_before is not None:
            win_mask, win_flag = self._getWindowMask(field, mask, flag_after, flag_before, win_flag, flag, force)
            out._flags.aloc[win_mask, field] = win_flag

        return out

    def _getWindowMask(self, field, mask, flag_after, flag_before, win_flag, flag, force):
        """ Return a mask which is True where the additional window flags should get set.

        Parameters
        ----------
        field : str
            column identifier.
        mask : boolean pd.Series
            identified locations where flags was set
        flag_after : offset or int
            set additional flags after each flag that was set
        flag_before : offset or int
            set additional flags before each flag that was set
        win_flag : any
            Should be valid flag of the flagger or None. Defaults to `flag` if None.
        flag : any
            The flag that was used by flagger.setFlags(). Only used to determine `win_flag` if the latter is None.
        force : bool
            If True, the additional flags specified by `flag_after` and `flag_before` are set unconditionally and so
            also could overwrite worse flags.

        Returns
        -------
        mask: boolean pandas.Series
            locations where additional flags should be set. The mask has the same (complete) length than `.flags[field]`
        win_flag: the flag to set

        Raises
        ------
        ValueError : If `win_flag` is None and `flag` is not a scalar.
        ValueError : If `win_flag` is not a valid flagger flag
        NotImplementedError: if `flag_before` is given
        """

        # win_flag default to flag if not explicitly given
        if win_flag is None:
            win_flag = flag
            if not np.isscalar(win_flag):
                raise ValueError("win_flag (None) cannot default to flag, if flag is not a scalar. "
                                 "Pls specify `win_flag` or omit `flag_after` and `flag_before`.")
        else:
            if not self.isValidFlag(win_flag):
                raise ValueError(f"invalid win_flag: {win_flag}")

        # blow up the mask to the whole size of flags
        base = mask.reindex_like(self._flags[field]).fillna(False)
        before, after = False, False

        if flag_after is not None:
            if isinstance(flag_after, int):
                flag_after += 1
            after = base.rolling(window=flag_after, min_periods=1, closed='both').sum().astype(bool)

        if flag_before is not None:
            raise NotImplementedError("flag_before is not implemented")

        # does not include base, to avoid overriding flags that just was set
        # by the test, because flag and win_flag may differ.
        mask = ~base & (after | before)

        # also do not to overwrite worse flags
        if not force:
            mask &= self.getFlags(field) < win_flag

        return mask, win_flag

    def clearFlags(self, field: str, loc: LocT = None, inplace=False, **kwargs) -> BaseFlaggerT:
        assertScalar("field", field, optional=False)
        if "force" in kwargs:
            raise ValueError("Keyword 'force' is not allowed here.")
        if "flag" in kwargs:
            raise ValueError("Keyword 'flag' is not allowed here.")
        return self.setFlags(field=field, loc=loc, flag=self.UNFLAGGED, force=True, inplace=inplace, **kwargs)

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

        # notna() to prevent nans to become True,
        # eg.: `np.nan != 0 -> True`
        flagged = flags.notna()

        # passing an empty list must result
        # in a everywhere-False data
        if len(flags_to_compare) == 0:
            flagged[:] = False
        else:
            for f in flags_to_compare:
                if not self.isValidFlag(f):
                    raise ValueError(f"invalid flag: {f}")
                flagged &= cp(flags, f)

        return flagged

    def copy(self, flags=None) -> BaseFlaggerT:
        if flags is None:
            out = deepcopy(self)
        else:
            # if flags is given and self.flags is big,
            # this hack will bring some speed improvement
            # NOTE: there should be nicer way to do this,
            #       why not through a constructur method?
            saved = self._flags
            self._flags = None
            out = deepcopy(self)
            out._flags = flags.copy()
            self._flags = saved
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

    def replaceField(self, field, flags, inplace=False, **kwargs):
        """ Replace or delete all data for a given field.

        Parameters
        ----------
        field : str
            The field to replace / delete. If the field already exist, the respected data
            is replaced, otherwise the data is inserted in the respected field column.
        flags : pandas.Series or None
            If None, the series denoted by `field` will be deleted. Otherwise
            a series of flags (dtype flagger.dtype) that will replace the series
            currently stored under `field`
        inplace : bool, default False
            If False, a flagger copy is returned, otherwise the flagger is not copied.
        **kwargs : dict
            ignored.

        Returns
        -------
        flagger: saqc.flagger.BaseFlagger
            The flagger object or a copy of it (if inplace=True).

        Raises
        ------
        ValueError: (delete) if field does not exist
        TypeError: (replace / insert) if flags are not pd.Series
        """

        assertScalar("field", field, optional=False)

        out = self if inplace else deepcopy(self)

        # delete
        if flags is None:
            if field not in self._flags:
                raise ValueError(f"{field}: field does not exist")
            del out._flags[field]

        # insert / replace
        else:
            if not isinstance(flags, pd.Series):
                raise TypeError(f"`flags` must be pd.Series.")
            out._flags[field] = flags.astype(self.dtype)
        return out

    def _check_field(self, field):
        """ Check if (all) field(s) in self._flags. """

        # wait for outcome of
        # https://git.ufz.de/rdm-software/saqc/issues/46
        failed = []
        if isinstance(field, str):
            if field not in self._flags:
                failed += [field]
        else:
            try:
                for f in field:
                    if f not in self._flags:
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
