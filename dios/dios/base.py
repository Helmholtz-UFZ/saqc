#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy, copy as shallowcopy
from typing import Mapping, Hashable, Any, Sequence, overload

from . import operators as ops
from . import pandas_bridge as pdextra
from . import lib

from .lib import (
    _CAST_POLICIES,
    _throw_MixedItype_err_or_warn,
    _find_least_common_itype,
)

from abc import abstractmethod
import pandas as pd
import operator as op
import functools as ftools

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"


class _DiosBase:
    @property
    @abstractmethod
    def _constructor(self) -> type[_DiosBase]:
        raise NotImplementedError

    def _finalize(self, other: _DiosBase):
        self._attrs = other._attrs
        return self

    def __init__(
        self,
        data=None,
        columns=None,
        index=None,
        itype=None,
        cast_policy="save",
        fastpath=False,
    ):

        self._attrs = {}
        self.cast_policy = cast_policy  # set via property

        # we are called internally
        if fastpath:
            self._itype = itype or lib.ObjItype
            if data is not None:
                self._data = data
            else:
                # it is significantly faster, to provide an index and fill it,
                # than to successively build the index by adding data
                self._data = pd.Series(dtype="O", index=columns)

        else:

            if index is not None and not isinstance(index, pd.Index):
                index = pd.Index(index)

            # itype=None means infer the itype by the data, so we first set to the highest
            # possible itype, then insert data, then infer the best-fitting itype.
            if itype is None and index is None:
                self._itype = lib.ObjItype
            else:
                if index is not None:
                    self._itype = lib.get_itype(index)
                if itype is not None:
                    self._itype = lib.get_itype(itype)

            cols = pd.Index([] if columns is None else columns)
            if not cols.is_unique:
                raise ValueError("columns must be unique")
            self._data = pd.Series(dtype="O", index=cols)

            if data is not None:
                self._init_insert_data(data, columns, index)

        # self._data may still contain nans; at positions where
        # no data was present, but a column-name was given
        if self._data.hasnans:
            e = pd.Series(dtype="O", index=index)
            for c in self.columns[self._data.isna()]:
                self._insert(c, e.copy())

        self._data.index.name = "columns"

        # we try to infer the itype, but if we still have
        # no data, we will set the itype lazy, i.e. with
        # the first non-empty _insert()
        if itype is None:
            if self.empty:
                self._itype = "INFER"
            else:
                self._itype = _find_least_common_itype(self._data)
                if not self._itype.unique:
                    _throw_MixedItype_err_or_warn(self.itype)

    def _init_insert_data(self, data, columns, index):
        """Insert items of a iterable in self"""

        if pdextra.is_iterator(data):
            data = list(data)

        if _is_dios_like(data) and not data.columns.is_unique:
            raise ValueError("columns index must have unique values")

        if _is_dios_like(data) or isinstance(data, dict):
            if columns is None:
                pass  # data is dict-like
            else:
                data = {k: data[k] for k in data if k in columns}

        elif isinstance(data, pd.Series):
            name = data.name or 0
            if columns is not None and len(columns) > 0:
                name = self.columns[0]
            data = {name: data}

        elif pdextra.is_nested_list_like(data):
            if columns is None:
                data = {i: d for i, d in enumerate(data)}
            elif len(data) == len(columns):
                data = dict(zip(self.columns, data))
            else:
                raise ValueError(
                    f"{len(columns)} columns passed, data implies {len(data)} columns"
                )

        elif pdextra.is_list_like(data):
            name = 0 if columns is None or len(columns) < 1 else self.columns[0]
            data = {name: data}

        else:
            raise TypeError("data type not understood")

        for k in data:
            s = pd.Series(data[k], index=index, dtype=object).infer_objects()
            self._insert(k, s)

    # ----------------------------------------------------------------------
    # checks

    def _is_valid_columns_index(self, obj):
        if isinstance(obj, pd.Series) and obj.dtype == "O":
            return True
        return False

    # ----------------------------------------------------------------------
    # Indexing Methods

    def _insert(self, col, val):
        """Insert a fresh new value as pd.Series into self"""
        val = list(val) if pdextra.is_iterator(val) else val

        if _is_dios_like(val):
            val = val.squeeze()
            if not isinstance(val, pd.Series):
                raise ValueError(f"Cannot insert frame-like with more than one column")

        elif val is None:
            val = pd.Series()

        elif not isinstance(val, pd.Series):
            raise TypeError(
                f"Only data of type pandas.Series can be inserted, passed was {type(val)}"
            )

        # set the itype lazy, i.e. when first non-empty
        # column is inserted
        if self._itype == "INFER":
            if not val.empty:
                self._itype = lib.get_itype(val.index)
                # cast all pre-inserted empty series
                self._cast_all(self._itype, self.cast_policy)
                if not self._itype.unique:
                    _throw_MixedItype_err_or_warn(self._itype)
        else:
            val = lib.cast_to_itype(val, self.itype, policy=self.cast_policy)

        val.name = col
        self._data.at[col] = val.copy(deep=True)

    @overload
    def __getitem__(self, key: str | int) -> pd.Series:
        ...

    @overload
    def __getitem__(self, key: slice) -> pd.Series:
        ...

    @overload
    def __getitem__(self, key: "_DiosBase" | pd.DataFrame) -> "_DiosBase":
        ...

    @overload
    def __getitem__(self, key: Sequence[str | int]) -> "_DiosBase":
        ...

    def __getitem__(self, key):
        """dios[key] -> dios/series"""
        # scalar        -> select a column
        # slice         -> select rows (on all columns)
        # bool dios     -> select columns, select rows
        # mask          -> select rows (on all columns)
        # list-like     -> select columns

        if pdextra.is_scalar(key):
            # NOTE: we shallow copy, to prevent changes on the
            # index mirror back to us and may mess up the itype.
            s = self._data.at[key]
            s.index = s.index.copy()
            return s

        if isinstance(key, slice):
            return self._slice(key)

        if _is_dios_like(key):
            return self._getitem_bool_dios(key)

        if pdextra.is_bool_indexer(key):
            return self._getitem_bool_listlike(key)

        # select columns and let pandas handle it
        data = self._data.loc[key]
        if self._is_valid_columns_index(data):
            return self._constructor(
                data=data, itype=self.itype, cast_policy=self.cast_policy, fastpath=True
            )._finalize(self)

        raise TypeError(f"cannot index columns with this type, {type(key)}")

    def _slice(self, key):
        """slices self, return copy"""
        if key == slice(None):
            return self.copy()

        new = self.copy_empty(columns=True)
        for c, series in self.items():
            new._data.at[c] = series[key]
        return new

    def _getitem_bool_dios(self, key):
        """Select items by a boolean dios-like drop un-selected indices."""

        if not _is_bool_dios_like(key):
            raise ValueError("Must pass DictOfSeries with boolean values only")

        new = self.copy_empty(columns=True)
        for c, series in self.items():
            if c in key:
                val = key[c].reindex(index=series.index, fill_value=False)
                new._data.at[c] = series.loc[val]
        return new

    def _getitem_bool_listlike(self, key):
        new = self.copy_empty(columns=True)
        for c, series in self.items():
            new._data.at[c] = series.loc[key]
        return new

    def __setitem__(self, key, value):
        """dios[key] = value"""
        key = list(key) if pdextra.is_iterator(key) else key
        if isinstance(key, tuple):
            raise KeyError(f"{key}. tuples are not allowed")

        elif pdextra.is_hashable(key):
            if isinstance(value, pd.Series) or key not in self.columns:
                self._insert(key, value)
            elif _is_dios_like(value) or pdextra.is_nested_list_like(value):
                raise ValueError("Incompatible indexer with multi-dimensional value")
            else:
                self._data.at[key][:] = value

        else:
            data = self.__getitem__(key)
            assert isinstance(
                data, self.__class__
            ), f"getitem returned data of type {type(data)}"

            # special cases
            if _is_dios_like(value):
                self._setitem_dios(data, value)
            # NOTE: pd.Series also considered list-like
            elif pdextra.is_list_like(value):
                self._setitem_listlike(data, value)

            # default case
            else:
                for c, series in data.items():
                    series[:] = value
                    self._data.at[c][series.index] = series

    def _setitem_listlike(self, data, value):

        value = value.values if isinstance(value, pd.Series) else value

        if len(value) != len(data.columns):
            raise ValueError(
                f"array-like value of length {len(value)} could "
                f"not be broadcast to indexing result of shape "
                f"(.., {len(data.columns)})"
            )

        for i, (c, series) in enumerate(data.items()):
            series[:] = value[i]
            self._data.at[c][series.index] = series

    def _setitem_dios(self, data, value):
        """Write values from a dios-like to self.

        No justification or alignment of columns, but of indices.
        If value has missing indices, nan's are inserted at that
        locations, just like `series.loc[:]=val` or `df[:]=val` do.

        Eg.
         di[::2] = di[::3]   ->   di[::2]

            x |        x |            x |
        ===== |     ==== |       ====== |
        0   x |     0  z |       0    z |
        2   x |  =  3  z |   ->  2  NaN |
        4   x |     6  z |       4  NaN |
        6   x |                  6    z |

        Parameter
        ----------
        data : dios
            A maybe trimmed version of self
        value : dios, pd.Dataframe
            The value to set with the same column dimension like data
        """

        if len(data) != len(value.columns):
            raise ValueError(
                f"shape mismatch: values array of shape "
                f"(.., {len(value.columns)}) could not "
                f"be broadcast to indexing result of "
                f"shape (.., {len(data.columns)})"
            )

        for i, (c, series) in enumerate(data.items()):
            # .loc cannot handle empty series,
            # like `emptySeries.loc[:] = [1,2]`
            if series.empty:
                continue
            val = value[value.columns[i]]
            series.loc[:] = val
            self._data.at[c].loc[series.index] = series

    def __delitem__(self, key):
        del self._data[key]

    # ------------------------------------------------------------------------------
    # Base properties and basic dunder magic

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.
        """
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def columns(self):
        """The column labels of the DictOfSeries"""
        return self._data.index

    @columns.setter
    def columns(self, cols):
        index = pd.Index(cols)
        if not index.is_unique:
            raise ValueError("columns index must have unique values")
        self._data.index = index
        # rename all columns
        for i, s in enumerate(self._data):
            s.name = index[i]

    @property
    def itype(self):
        """The ``Itype`` of the DictOfSeries.

        See :ref:`Itype documentation <doc_itype:Itype>` for more info.
        """
        if self._itype == "INFER":
            return None
        return self._itype

    @itype.setter
    def itype(self, itype):
        itype = lib.get_itype(itype)
        self._cast_all(itype, policy=self.cast_policy)
        self._itype = itype

    @property
    def cast_policy(self):
        """The policy to use for casting new columns if its initial itype does not fit.

        See :ref:`Itype documentation <doc_itype:Itype>` for more info.
        """
        return self._policy

    @cast_policy.setter
    def cast_policy(self, policy):
        if policy not in _CAST_POLICIES:
            raise ValueError(f"policy must be one of {_CAST_POLICIES}")
        self._policy = policy

    def _cast_all(self, itype, policy):
        c = "?"
        new = self.copy_empty()
        try:
            for c, series in self.items():
                new._data.at[c] = lib.cast_to_itype(series, itype, policy=policy)
        except Exception as e:
            raise type(e)(f"Column {c}: " + str(e)) from e

    def __len__(self):
        return len(self.columns)

    @property
    def empty(self):
        """Indicator whether DictOfSeries is empty.

        Returns
        -------
        bool :
            If DictOfSeries is empty, return True, if not return False.

        See Also
        --------
        DictOfSeries.dropempty : drop empty columns
        DictOfSeries.dropna : drop NAN's from a DictOfSeries
        pandas.Series.dropna : drop NAN's from a Series

        Notes
        -----
            If DictOfSeries contains only NaNs, it is still not considered empty. See the example below.

        Examples
        --------
        An example of an actual empty DictOfSeries.

        >>> di_empty = DictOfSeries(columns=['A'])
        >>> di_empty
        Empty DictOfSeries
        Columns: ['A']
        >>> di_empty.empty
        True

        If we only have NaNs in our DictOfSeries, it is not considered empty!
        We will need to drop the NaNs to make the DictOfSeries empty:

        >>> di = pd.DictOfSeries({'A' : [np.nan]})
        >>> di
            A |
        ===== |
        0 NaN |
        >>> di.empty
        False
        >>> di.dropna().empty
        True
        """
        return len(self) == 0 or all(s.empty for s in self._data)

    def __iter__(self):
        yield from self.columns

    def __reversed__(self):
        yield from reversed(self.columns)

    def __contains__(self, item):
        return item in self.columns

    def items(self):
        yield from self._data.items()

    # ----------------------------------------------------------------------
    # if copy.copy() is copy.copy(): return copy.copy().copy()

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True)

    def __copy__(self):
        return self.copy(deep=False)

    def copy(self, deep=True):
        """Make a copy of this DictOfSeries' indices and data.

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With deep=False neither the indices nor the data are copied.

        Returns
        -------
        copy : DictOfSeries

        See Also
        --------
        pandas.DataFrame.copy
        """
        data = self._data.copy()  # always copy the outer hull series
        if deep:
            for c, series in self.items():
                data.at[c] = series.copy()

        new = self._constructor(
            data=data, itype=self.itype, cast_policy=self.cast_policy, fastpath=True
        )

        copyfunc = deepcopy if deep else shallowcopy
        new._attrs = copyfunc(self._attrs)

        return new

    def copy_empty(self, columns=True):
        """
        Return a new DictOfSeries object, with same properties than the original.
        Parameters
        ----------
        columns: bool, default True
             If ``True``, the copy will have the same, but empty columns like the original.

        Returns
        -------
        DictOfSeries: empty copy

        Examples
        --------

        >>> di = DictOfSeries({'A': range(2), 'B': range(3)})
        >>> di
           A |    B |
        ==== | ==== |
        0  0 | 0  0 |
        1  1 | 1  1 |
             | 2  2 |

        >>> empty = di.copy_empty()
        >>> empty
        Empty DictOfSeries
        Columns: ['A', 'B']

        The properties are the same, eg.

        >>> empty.itype == di.itype
        True
        >>> empty.cast_policy == di.cast_policy
        True
        >>> empty.dtypes == di.dtypes
        columns
        A    True
        B    True
        dtype: bool
        """
        data = None
        if columns is True:  # is correct
            data = pd.Series(dtype="O", index=self.columns)
            for c, series in self.items():
                # OPTIM: the following code is about 2x faster than
                # data.at[c] = pd.Series(dtype=self._data.at[c].dtype)
                data.at[c] = series.reindex([])

        return self._constructor(
            data=data, itype=self.itype, cast_policy=self.cast_policy, fastpath=True
        )._finalize(self)

    # ------------------------------------------------------------------------------
    # Operators

    def _op1(self, op):
        new = self.copy_empty(columns=True)
        try:
            for k, series in self.items():
                new[k] = op(series)
        except Exception as e:
            raise type(e)(f"'{ops.OP_MAP[op]} dios' failed: " + str(e)) from e
        return new

    def _op2_inplace(self, op, other, align=True) -> None:
        new = self._op2(op, other, align)
        self._data = new._data

    def _op2(self, op, other, align=True) -> "_DiosBase":
        def raiseif(kself, kother, s):
            if kself != kother:
                raise ValueError(
                    f"{s} does not match, {s} left: {kself}, {s} right: {kother}"
                )

        def doalign(left, right):
            return left.align(right, join="inner") if align else (left, right)

        def get_operants():
            if _is_dios_like(other):
                raiseif(list(self), list(other), "keys")
                for k, series in self.items():
                    yield (k, *doalign(series, other[k]))
            elif isinstance(other, pd.Series):
                for k, series in self.items():
                    yield (k, *doalign(series, other))
            elif pdextra.is_dict_like(other):
                raiseif(sorted(self), sorted(other), "keys")
                for k, series in self.items():
                    yield (k, series, other[k])
            elif pdextra.is_nested_list_like(other):
                raiseif(len(self), len(other), "length")
                for i, (k, series) in enumerate(self.items()):
                    yield (k, series, other[i])
            elif pdextra.is_scalar(other) or pdextra.is_list_like(other):
                for k, series in self.items():
                    yield (k, series, other)
            else:
                raise NotImplementedError

        new = self.copy_empty(columns=True)
        try:
            for k, ser, oth in get_operants():
                new[k] = op(ser, oth)
        except Exception as e:
            raise type(e)(f"'dios {ops.OP_MAP[op]} other' failed: " + str(e)) from e

        return new

    # unary
    __neg__ = ftools.partialmethod(_op1, op.neg)
    __abs__ = ftools.partialmethod(_op1, op.abs)
    __invert__ = ftools.partialmethod(_op1, op.inv)
    # comparison
    __eq__ = ftools.partialmethod(_op2, op.eq, align=False)
    __ne__ = ftools.partialmethod(_op2, op.ne, align=False)
    __le__ = ftools.partialmethod(_op2, op.le, align=False)
    __ge__ = ftools.partialmethod(_op2, op.ge, align=False)
    __lt__ = ftools.partialmethod(_op2, op.lt, align=False)
    __gt__ = ftools.partialmethod(_op2, op.gt, align=False)
    # arithmetic
    __add__ = ftools.partialmethod(_op2, op.add)
    __sub__ = ftools.partialmethod(_op2, op.sub)
    __mul__ = ftools.partialmethod(_op2, op.mul)
    __mod__ = ftools.partialmethod(_op2, op.mod)
    __truediv__ = ftools.partialmethod(_op2, op.truediv)
    __floordiv__ = ftools.partialmethod(_op2, op.floordiv)
    __pow__ = ftools.partialmethod(_op2, op.pow)
    __iadd__ = ftools.partialmethod(_op2_inplace, op.add)
    __isub__ = ftools.partialmethod(_op2_inplace, op.sub)
    __imul__ = ftools.partialmethod(_op2_inplace, op.mul)
    __imod__ = ftools.partialmethod(_op2_inplace, op.mod)
    __itruediv__ = ftools.partialmethod(_op2_inplace, op.truediv)
    __ifloordiv__ = ftools.partialmethod(_op2_inplace, op.floordiv)
    __ipow__ = ftools.partialmethod(_op2_inplace, op.pow)
    # bool
    __and__ = ftools.partialmethod(_op2, op.and_)
    __or__ = ftools.partialmethod(_op2, op.or_)
    __xor__ = ftools.partialmethod(_op2, op.xor)
    __iand__ = ftools.partialmethod(_op2, op.and_, inplace=True)
    __ior__ = ftools.partialmethod(_op2, op.or_, inplace=True)
    __ixor__ = ftools.partialmethod(_op2, op.xor, inplace=True)

    # ------------------------------------------------------------------------------
    # Indexer

    @property
    def loc(self):
        """Access a group of rows and columns by label(s) or a boolean array.

        See :ref:`indexing docs <doc_indexing:Pandas-like indexing>`
        """
        return _LocIndexer(self)

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        See :ref:`indexing docs <doc_indexing:Pandas-like indexing>`
        """
        return _iLocIndexer(self)

    @property
    def aloc(self):
        """Access a group of rows and columns by label(s) or a boolean array with automatic alignment of indexers.

        See :ref:`indexing docs <doc_indexing:Special indexer .aloc>`
        """
        return _aLocIndexer(self)

    @property
    def at(self):
        """Access a single value for a row/column label pair.

        See :ref:`indexing docs <doc_indexing:Pandas-like indexing>`
        """
        return _AtIndexer(self)

    @property
    def iat(self):
        """Access a single value for a row/column pair by integer position.

        See :ref:`indexing docs <doc_indexing:Pandas-like indexing>`
        """
        return _iAtIndexer(self)


def _is_dios_like(obj) -> bool:
    # must have columns
    # columns is some kind of pd.Index
    # iter will iter through columns
    # a `in` obj check if obj is in columns
    # obj[key] will give a pd.Series
    # obj.squeeze() give pd.Series if len(obj) == 1
    return isinstance(obj, (_DiosBase, pd.DataFrame))


def _is_bool_series(obj) -> bool:
    return isinstance(obj, pd.Series) and obj.dtype == bool


def _is_bool_dios_like(obj) -> bool:
    if not _is_dios_like(obj):
        return False
    dtypes = obj.dtypes
    if (dtypes == bool).all():
        return True
    if (dtypes == "O").any():
        return obj.apply(pdextra.is_bool_indexer).all()
    return False


# keep this here to prevent cyclic import
from .indexer import _aLocIndexer, _iLocIndexer, _LocIndexer, _iAtIndexer, _AtIndexer
