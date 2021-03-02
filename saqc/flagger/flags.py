#!/usr/bin/env python

from __future__ import annotations

import dios
from saqc.common import *
from saqc.flagger.history import History
import pandas as pd
from typing import Union, Dict, DefaultDict, Optional, Type, Tuple, Iterable

_VAL = Union[pd.Series, History]
DictLike = Union[
    pd.DataFrame,
    dios.DictOfSeries,
    Dict[str, _VAL],
    DefaultDict[str, _VAL],
]

_Field = str
SelectT = Union[
    _Field,
    Tuple[pd.Series, _Field]
]
ValueT = Union[pd.Series, Iterable, float]


class _HistAccess:

    def __init__(self, obj: Flags):
        self.obj = obj

    def __getitem__(self, key: str) -> History:
        # we don't know, what the user wants. Although we're not
        # encouraging inplace modification of the history, the
        # user may do it, so we remove the cached column here.
        self.obj._cache.pop(key, None)
        return self.obj._data[key]

    def __setitem__(self, key: str, value: Union[History, pd.DataFrame]):
        if not isinstance(value, History):
            value = History(value)
        self.obj._data[key] = value
        self.obj._cache.pop(key, None)


class Flags:
    """
    flags manipulation
    ------------------
    insert new    -> flags['new'] = pd.Series(...)
    set items     -> flags['v1'] = pd.Series(...)
    get items     -> v0 = flags['v0']
    delete items  -> del flags['v0']  / drop('v0')

    metadata
    --------
    reading columns     -> flags.columns
    renaming column(s)  -> flags.columns = pd.Index(['a', 'b', 'c'])

    history
    -------
    get history  -> flags.history['v0']
    set history  -> flags.history['v0'] = History(...)

    conversion
    ----------
    make a dios  -> flags.toDios()
    make a df    -> flags.toFrame()
    """

    def __init__(self, raw_data: Optional[Union[DictLike, Flags]] = None, copy: bool = False):

        if raw_data is None:
            raw_data = {}

        if isinstance(raw_data, Flags):
            raw_data = raw_data._data

        # with python 3.7 dicts are insertion-ordered by default
        self._data = self._initFromRaw(raw_data, copy)

        # this is a simple cache that reduce the calculation of the flags
        # from the entire history of a flag column. The _cache is filled
        # with __getitem__ and cleared on any write access to self_data.
        # There are not to may write access possibilities here so we don't
        # have to much trouble.
        self._cache = {}

    def _initFromRaw(self, data, copy) -> Dict[str, History]:
        """
        init from dict-like: keys are flag column, values become
        initial columns of history(s).
        """
        result = {}

        for k, item in data.items():

            if k in result:
                raise ValueError('raw_data must not have duplicate keys')

            # No, means no ! (copy)
            if isinstance(item, History) and not copy:
                result[k] = item
                continue

            if isinstance(item, pd.Series):
                item = item.to_frame(name=0)
            elif isinstance(item, History):
                pass
            else:
                raise TypeError(f"cannot init from {type(data.__name__)} of {type(item.__name__)}")

            result[k] = History(item, copy=copy)

        return result

    @property
    def _constructor(self) -> Type['Flags']:
        return type(self)

    # ----------------------------------------------------------------------
    # mata data

    @property
    def columns(self) -> pd.Index:
        return pd.Index(self._data.keys())

    @columns.setter
    def columns(self, value: pd.Index):
        if not isinstance(value, pd.Index):
            value = pd.Index(value)

        if (
                not value.is_unique
                or not pd.api.types.is_string_dtype(value)
        ):
            raise TypeError('value must be pd.Index, with unique indices of type str')

        if not len(value) == len(self):
            raise ValueError("index must match current index in length")

        _data, _cache = {}, {}

        for old, new in zip(self.columns, value):
            _data[new] = self._data[old]

            if old in self._cache:
                _cache[new] = self._cache[old]

        self._data = _data
        self._cache = _cache

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item):
        return item in self.columns

    # ----------------------------------------------------------------------
    # item access

    def __getitem__(self, key: str) -> pd.Series:

        if key not in self._cache:
            self._cache[key] = self._data[key].max()

        return self._cache[key].copy()

    def __setitem__(self, key: SelectT, value: ValueT):
        # force-KW is internal available only

        if isinstance(key, tuple):
            if len(key) != 2:
                raise KeyError("a single 'column' or a tuple of 'mask, column' must be passt")
            mask, key = key

            # raises (correct) KeyError
            tmp = pd.Series(UNTOUCHED, index=self._data[key].index, dtype=float)
            try:
                tmp[mask] = value
            except Exception:
                raise ValueError('bad mask')
            else:
                value = tmp

        # technically it would be possible to select a field and set
        # the entire column to a scalar flag value (float), but it has
        # a high potential, that this is not intended by the user.
        # if desired use ``flagger[:, field] = flag``
        if not isinstance(value, pd.Series):
            raise ValueError("must pass value of type pd.Series")

        # if nothing happens no-one writes the history books
        if len(value) == 0:
            return

        if key not in self._data:
            self._data[key] = History()

        self._data[key].append(value, force=True)
        self._cache.pop(key, None)

    def __delitem__(self, key):
        self._data.pop(key)
        self._cache.pop(key, None)

    def drop(self, key: str):
        """
        Delete a flags column.

        Parameters
        ----------
        key : str
            column name

        Returns
        -------
        Flags
            the same flags object with dropeed column, no copy
        """
        self.__delitem__(key)

    # ----------------------------------------------------------------------
    # accessor

    @property
    def history(self) -> _HistAccess:
        return _HistAccess(self)

    # ----------------------------------------------------------------------
    # copy

    def copy(self, deep=True):
        return self._constructor(self, copy=deep)

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    # ----------------------------------------------------------------------
    # transformation and representation

    def toDios(self) -> dios.DictOfSeries:
        di = dios.DictOfSeries(columns=self.columns)

        for k, v in self._data.items():
            di[k] = self[k]  # use cache

        return di.copy()

    def toFrame(self) -> pd.DataFrame:
        return self.toDios().to_df()

    def __repr__(self) -> str:
        return str(self.toDios()).replace('DictOfSeries', type(self).__name__)


def initFlagsLike(reference: Union[pd.Series, DictLike, Flags], initial_value: float = UNFLAGGED) -> Flags:
    """
    Create empty Flags, from an reference data structure.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, dios.DictOfSeries, dict of pd.Series
        The reference structure to initialize for.

    initial_value : float, default 0
        value to initialize the columns with

    Notes
    -----
    Implementation detail:

    The resulting Flags has not necessarily the exact same (inner) dimensions as the reference.
    This may happen, if the passed structure, already holds History objects. Those are
    reduced 1D-DataFrame (1-column-History). Nevertheless the returned flags are perfectly suitable
    to be used in Saqc as flags container along with the passed reference structure (data).

    Returns
    -------
    flags: Flags
        a flags object,
    """
    result = {}

    if isinstance(reference, Flags):
        reference = reference._data

    if isinstance(reference, pd.Series):
        reference = reference.to_frame('f0')

    for k, item in reference.items():

        if not isinstance(k, str):
            raise TypeError(f"cannot use {k} as key, currently only string keys are allowed")

        if k in result:
            raise ValueError('reference must not have duplicate keys')

        if not isinstance(item, (pd.Series, History)):
            raise TypeError('items in reference must be of type pd.Series')

        item = pd.DataFrame(initial_value, index=item.index, columns=[0], dtype=float)

        result[k] = History(item)

    return Flags(result)


# for now we keep this name
Flagger = Flags
