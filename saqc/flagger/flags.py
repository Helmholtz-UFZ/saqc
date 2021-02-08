#!/usr/bin/env python

from __future__ import annotations

import dios

from saqc.flagger.history import History
import numpy as np
import pandas as pd
from typing import Union, Dict, DefaultDict, Iterable, Tuple, Optional

UNTOUCHED = np.nan
UNFLAGGED = 0
DOUBTFUL = 25
BAD = 99

_KEY = str
_VAL = Union[pd.Series, History]
DictLike = Union[
    pd.DataFrame,
    dios.DictOfSeries,
    Dict[_KEY, _VAL],
    DefaultDict[_KEY, _VAL],
    Iterable[Tuple[_KEY, _VAL]]
]


class _HistAccess:

    def __init__(self, obj: Flags):
        self.obj = obj

    def __getitem__(self, key: str) -> History:
        return self.obj._data[key]

    def __setitem__(self, key: str, value: Union[History, pd.DataFrame]):
        if not isinstance(value, History):
            value = History(value)
        self.obj._data[key] = value
        self.obj._cache.clear()


class Flags:

    def __init__(self, raw_data: Optional[Union[DictLike, Flags]] = None, copy: bool = False):

        if raw_data is None:
            raw_data = {}

        if isinstance(raw_data, Flags):
            raw_data = raw_data._data

        self._data: Dict[str, History]
        self._data = self._init_from_raw(raw_data, copy)

        # this is a simple cache that reduce the calculation of the flags
        # from the entire history of a flag column. The _cache is filled
        # with __getitem__ and cleared in __setitem__ or if the whole history
        # is written in _HistAccess.__setitem__. There is no other access, so
        # we don't have to much trouble.
        self._cache = {}

    def _init_from_raw(self, data, copy) -> Dict[str, History]:
        result = {}

        for obj in data:
            if isinstance(obj, tuple):
                k, item = obj
            else:
                k, item = obj, data[obj]

            if k in result:
                raise ValueError('raw_data must not have duplicate keys')

            if isinstance(item, pd.Series):
                item = item.to_frame(name=0)

            result[k] = History(item, copy=copy)

        return result

    def __getitem__(self, key: str) -> pd.Series:
        if key not in self._cache:
            self._cache[key] = self._data[key].max()
        return self._cache[key]

    def __setitem__(self, key: str, value: pd.Series):
        if key not in self._data:
            hist = History()
        else:
            hist = self._data[key]

        hist.append(value)
        self._cache.pop(key, None)

    @property
    def columns(self) -> pd.Index:
        return pd.Index(self._data.keys())

    @property
    def history(self) -> _HistAccess:
        return _HistAccess(self)

    def to_dios(self) -> dios.DictOfSeries:
        di = dios.DictOfSeries(columns=self.columns)

        for k, v in self._data.items():
            di[k] = self[k]  # cached

        return di.copy()

    def to_frame(self) -> pd.DataFrame:
        return self.to_dios().to_df()

    def __repr__(self) -> str:
        return str(self.to_dios())


def init_flags_like(reference: Union[pd.Series, DictLike, Flags]) -> Flags:
    """
    Create empty Flags, from an reference data structure.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, dios.DictOfSeries, dict of pd.Series
        The reference structure to initialize for.

    Returns
    -------

    """
    result = {}

    if isinstance(reference, Flags):
        reference = reference._data

    if isinstance(reference, pd.Series):
        reference = reference.to_frame('f0')

    for obj in reference:

        # unpack
        if isinstance(obj, tuple):
            k, item = obj
        else:
            k, item = obj, reference[obj]

        if not isinstance(k, str):
            raise TypeError(f"cannot use {k} as key, currently only string keys are allowed")

        if k in result:
            raise ValueError('reference must not have duplicate keys')

        if not isinstance(item, (pd.Series, History)):
            raise TypeError('items in reference must be of type pd.Series')

        item = pd.DataFrame(UNFLAGGED, index=item.index, columns=[0], dtype=float)

        result[k] = History(item)

    return Flags(result)


if __name__ == '__main__':
    from dios import example_DictOfSeries
    f = Flags(example_DictOfSeries().astype(float))
    print(f)