#!/usr/bin/env python

from __future__ import annotations

import pandas as pd
import dios
from typing import Union, Dict, DefaultDict, Optional, Type, Tuple, Iterable

from saqc.constants import *
from saqc.core.history import History


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
    Tuple[pd.Series, _Field],
    Tuple[pd.Index, _Field],
    Tuple[slice, _Field],
]
ValueT = Union[pd.Series, Iterable, float]


class _HistAccess:
    def __init__(self, obj: Flags):
        self.obj = obj

    def __getitem__(self, key: str) -> History:
        return self.obj._data[key].copy()

    def __setitem__(self, key: str, value: Union[History, pd.DataFrame]):
        if not isinstance(value, History):
            value = History(value)

        if not isinstance(value, History):
            raise TypeError("Not a History")

        History._validateHistWithMask(value.hist, value.mask)

        self.obj._data[key] = value
        self.obj._cache.pop(key, None)


class Flags:
    """
    Saqc's flags container.

    This container class holds the quality flags associated with the data. It hold key-value pairs, where
    the key is the name of the column and the value is a ``pandas.Series`` of flags. The index of the series
    and the key-value pair can be assumed to be immutable, which means, only the *values* of the series can
    be change, once the series exist.
    In other words: **an existing column can not be overwritten by a column with a different index.**

    The flags can be accessed via ``__getitem__`` and ``__setitem__``, in real life known as the `[]`-operator.

    For the curious:
        Under the hood, the series are stored in a `history`, which allows the advanced user to retrieve all flags
        once was set in this object, but in the most cases this is irrelevant. For simplicity one can safely assume,
        that this class works just stores the flag-series one sets.

    See Also
    --------
    initFlagsLike : create a Flags instance, with same dimensions as a reference object.
    History : class that actually store the flags

    Examples
    --------
    We create an empty instance, by calling ``Flags`` without any arguments and then add a column to it.

    >>> from saqc.constants import UNFLAGGED, BAD, DOUBT, UNTOUCHED
    >>> flags = Flags()
    >>> flags
    Empty Flags
    Columns: []
    >>> flags['v0'] = pd.Series([BAD,BAD,UNFLAGGED], dtype=float)
    >>> flags
          v0 |
    ======== |
    0  255.0 |
    1  255.0 |
    2   -inf |

    Once the column exist, we cannot overwrite it anymore, with a different series.

    >>> flags['v0'] = pd.Series([666.], dtype=float)
    Traceback (most recent call last):
      some file path ...
    ValueError: Index does not match

    But if we pass a series, which index match it will work,
    because the series now is interpreted as value-to-set.

    >>> flags['v0'] = pd.Series([DOUBT,UNTOUCHED,DOUBT], dtype=float)
    >>> flags
          v0 |
    ======== |
    0   25.0 |
    1  255.0 |
    2   25.0 |

    As we see above, the column now holds a combination from the values from the
    first and the second set. This is, because the special constant ``UNTOUCHED``,
    an alias for ``numpy.nan`` was used. We can inspect all the updates that was
    made by looking in the history.

    >>> flags.history['v0']
            0       1
    0  (255.0)   25.0
    1   255.0     nan
    2   (-inf)   25.0

    As we see now, the second call sets ``25.0`` and shadows (represented by the parentheses) ``(255.0)`` in the
    first row and ``(-inf)`` in the last, but in the second row ``255.0`` still is valid, because it was
    `not touched` by the set.

    It is also possible to set values by a mask, which can be interpreted as condidional setting.
    Imagine we want to `reset` all flags to ``0.`` if the existing flags are lower that ``255.``.

    >>> mask = flags['v0'] < BAD
    >>> mask
    0     True
    1    False
    2     True
    dtype: bool
    >>> flags[mask, 'v0'] = 0
    >>> flags
          v0 |
    ======== |
    0    0.0 |
    1  255.0 |
    2    0.0 |

    The objects you can pass as a row selector (``flags[rows, column]``) are:

    - boolen arraylike, with or without index. Must have same length than the undeliing series.
    - slices working on the index
    - ``pd.Index``, which must be a subset of the existing index

    For example, to set `all` values to a scalar value, use a Null-slice:

    >>> flags[:, 'v0'] = 99.0
    >>> flags
         v0 |
    ======= |
    0  99.0 |
    1  99.0 |
    2  99.0 |

    After all calls presented here, the history look like this:

    >>> flags.history['v0']
            0       1      2       3
    0  (255.0)  (25.0)  (0.0)   99.0
    1  (255.0)   (nan)  (nan)   99.0
    2   (-inf)  (25.0)  (0.0)   99.0
    """

    def __init__(
        self, raw_data: Optional[Union[DictLike, Flags]] = None, copy: bool = False
    ):

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
                raise ValueError("raw_data must not have duplicate keys")

            # No, means no ! (copy)
            if isinstance(item, History) and not copy:
                result[k] = item
                continue

            if isinstance(item, pd.Series):
                item = item.to_frame(name=0)
            elif isinstance(item, History):
                pass
            else:
                raise TypeError(
                    f"cannot init from {type(data.__name__)} of {type(item.__name__)}"
                )

            result[k] = History(item, copy=copy)

        return result

    @property
    def _constructor(self) -> Type["Flags"]:
        return type(self)

    # ----------------------------------------------------------------------
    # mata data

    @property
    def columns(self) -> pd.Index:
        """
        Column index of the flags container

        Returns
        -------
        columns: pd.Index
            The columns index
        """
        return pd.Index(self._data.keys())

    @columns.setter
    def columns(self, value: pd.Index):
        """
        Set new columns names.

        Parameters
        ----------
        value : pd.Index
            New column names
        """
        if not isinstance(value, pd.Index):
            value = pd.Index(value)

        if not value.is_unique or not pd.api.types.is_string_dtype(value):
            raise TypeError("value must be pd.Index, with unique indices of type str")

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
        """
        True if flags has no columns.

        Returns
        -------
        bool
            ``True`` if the container has no columns, otherwise ``False``.
        """
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
                raise KeyError(
                    "a single 'column' or a tuple of 'mask, column' must be passt"
                )
            mask, key = key

            tmp = pd.Series(UNTOUCHED, index=self._data[key].index, dtype=float)

            # make a mask from an index, because it seems
            # that passing an index is a very common workflow
            if isinstance(mask, pd.Index):
                mask = pd.Series(True, index=mask, dtype=bool)
                mask = mask.reindex(tmp.index, fill_value=False)

            # raises (correct) KeyError
            try:
                tmp[mask] = value
            except Exception:
                raise ValueError("bad mask")
            else:
                value = tmp

        # technically it would be possible to select a field and set
        # the entire column to a scalar flag value (float), but it has
        # a high potential, that this is not intended by the user.
        # if desired use ``flags[:, field] = flag``
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
        flags object with dropped column, not a copy
        """
        self.__delitem__(key)

    # ----------------------------------------------------------------------
    # accessor

    @property
    def history(self) -> _HistAccess:
        """
        Accessor for the flags history.

        To get a copy of the current history use ``flags.history['var']``.
        To set a new history use ``flags.history['var'] = value``.
        The passed value must be a instance of History or must be convertible to a history.

        Returns
        -------
        history : History
            Accessor for the flags history

        See Also
        --------
        saqc.core.History : History storage class.
        """
        return _HistAccess(self)

    # ----------------------------------------------------------------------
    # copy

    def copy(self, deep=True):
        """
        Copy the flags container.

        Parameters
        ----------
        deep : bool, default True
            If False, a new reference to the Flags container is returned,
            otherwise the underlying data is also copied.

        Returns
        -------
        copy of flags
        """
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
        """
        Transform the flags container to a ``dios.DictOfSeries``.

        Returns
        -------
        dios.DictOfSeries
        """
        di = dios.DictOfSeries(columns=self.columns)

        for k, v in self._data.items():
            di[k] = self[k]  # use cache

        return di.copy()

    def toFrame(self) -> pd.DataFrame:
        """
        Transform the flags container to a ``pd.DataFrame``.

        Returns
        -------
        pd.DataFrame
        """
        return self.toDios().to_df()

    def __repr__(self) -> str:
        return str(self.toDios()).replace("DictOfSeries", type(self).__name__)


def initFlagsLike(
    reference: Union[pd.Series, DictLike, Flags],
    initial_value: float = UNFLAGGED,
    name: str = None,
) -> Flags:
    """
    Create empty Flags, from an reference data structure.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, dios.DictOfSeries, dict of pd.Series
        The reference structure to initialize for.

    initial_value : float, default 0
        value to initialize the columns with

    name : str, default None
        Only respected if `reference` is of type ``pd.Series``.
        The column name that is used for the Flags. If ``None``
        the name of the series itself is taken, if this is also
        `None`, a ValueError is raised.

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
        if name is None:
            name = reference.name
        if name is None:
            raise ValueError(
                "Either the passed series must be named or a name must be passed"
            )
        if not isinstance(name, str):
            raise TypeError(f"name must be str not '{type(name).__name__}'")
        reference = reference.to_frame(name=name)

    for k, item in reference.items():

        if not isinstance(k, str):
            raise TypeError(
                f"cannot use {k} as key, currently only string keys are allowed"
            )

        if k in result:
            raise ValueError("reference must not have duplicate keys")

        if not isinstance(item, (pd.Series, History)):
            raise TypeError("items in reference must be of type pd.Series")

        item = pd.DataFrame(initial_value, index=item.index, columns=[0], dtype=float)

        result[k] = History(item)

    return Flags(result)
