from __future__ import annotations
from typing import Any, Mapping, Hashable

from .base import _DiosBase, _is_dios_like
from .lib import Opts, OptsFields, dios_options
from .lib import _find_least_common_itype
from . import pandas_bridge as pdextra

import functools as ftools
import pandas as pd
import numpy as np


class DictOfSeries(_DiosBase):
    """A data frame where every column has its own index.

    DictOfSeries is a collection of pd.Series's which aim to be as close as possible similar to
    pd.DataFrame. The advantage over pd.DataFrame is, that every `column` has its own row-index,
    unlike the former, which provide a single row-index for all columns. This solves problems with
    unaligned data and data which varies widely in length.

    Indexing with ``di[]``, ``di.loc[]`` and ``di.iloc[]``  should work analogous to these methods
    from pd.DataFrame. The indexer can be a single label, a slice, a list-like, a boolean list-like,
    or a boolean DictOfSeries/pd.DataFrame and can be used to selectively get or set data.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series.

    columns : array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex(0, 1, 2, ..., n) if no column labels are provided.

    index : Index or array-like
        Index to use to reindex every given series during init. Ignored if omitted.

    itype : Itype, pd.Index, Itype-string-repr or type
        Every series that is inserted, must have an index of this type or any
        of this types subtypes.
        If None, the itype is inferred as soon as the first non-empty series is inserted.

    cast_policy : {'save', 'force', 'never'}, default 'save'
        Policy used for (down-)casting the index of a series if its type does not match
        the ``itype``.
    """

    @property
    def _constructor(self) -> type[DictOfSeries]:
        """Return the class. Useful for construction in the elder class.
        A import of DictOfSeries would end up cyclic."""
        return DictOfSeries

    def _construct_like_self(self, **kwargs):
        kwargs.setdefault("itype", self.itype)
        kwargs.setdefault("cast_policy", self.cast_policy)
        return self._constructor(**kwargs)._finalize(self)

    @property
    def indexes(self):
        """Return pandas.Series with the indexes of all columns."""
        return self.for_each("index")

    @property
    def values(self):
        """Return a numpy.array of numpy.arrays with the values of all columns.

        The outer has the length of columns, the inner holds the values of the column.
        """
        s = self.for_each("values")
        return s.values

    @property
    def dtypes(self):
        """Return pandas.Series with the dtypes of all columns."""
        return self.for_each("dtype")

    @property
    def lengths(self):
        """Return pandas.Series with the lenght of all columns."""
        return self._data.apply(len)

    @property
    def size(self):
        return self.lengths.sum()

    @property
    def shape(self):
        return tuple(self.lengths), len(self.columns)

    # ------------------------------------------------------------------------------
    # Dict-like methods

    def clear(self):
        d = self._data
        self._data = pd.Series(dtype=d.dtype, index=type(d.index)([]))

    def get(self, key, default=None):
        return self._data.get(key, default)

    # implemented in _BaseClass
    # def items(self):
    #     return self._data.items()

    def keys(self):
        return self.columns

    def pop(self, *args):
        # We support a default value, like dict, in contrary to pd.
        # Therefore we need to handle args manually, because dict-style pop()
        # differ between a single arg and a tuple-arg, with arg and default,
        # where the second arg can be anything, including None. If the key is
        # not present, and a single arg is given, a KeyError is raised, but
        # with a given default value, it is returned instead.
        if len(args) == 0:
            raise TypeError("pop expected at least 1 arguments, got 0")
        if len(args) > 2:
            raise TypeError(f"pop expected at most 2 arguments, got {len(args)}")
        key, *rest = args
        if key in self.columns:
            return self._data.pop(key)
        elif rest:
            return rest.pop()
        raise KeyError(key)

    def popitem(self):
        last = self.columns[-1]
        return last, self._data.pop(last)

    def setdefault(self, key, default=None):
        if key not in self.columns:
            self._insert(key, default)
        return self._data[key]

    def update(self, other):
        if not _is_dios_like(other):
            other = to_dios(other)
        self.aloc[other, ...] = other

    # ------------------------------------------------------------------------------
    # High-Level Iteration

    def iteritems(self):
        yield from self.items()

    def iterrows(self, fill_value=np.nan, squeeze=True):
        """
        Iterate over DictOfSeries rows as (index, pandas.Series/DictOfSeries) pairs.
        **MAY BE VERY PERFORMANCE AND/OR MEMORY EXPENSIVE**

        Parameters
        ----------
        fill_value: scalar, default numpy.nan
            Fill value for row entry, if the column does not have an entry
            at the current index location. This ensures that the returned
            Row always contain all columns. If ``None`` is given no value
            is filled.

            If ``fill_value=None`` and ``squeeze=True`` the resulting Row
            (a pandas.Series) may differ in length between iterator calls.
            That's because an entry, that is not present in a column, will
            also not be present in the resulting Row.

        squeeze: bool, default False
            * ``True`` : A pandas.Series is returned for each row.
            * ``False`` : A single-rowed DictOfSeries is returned for each row.

        Yields
        ------
        index : label
            The index of the row.
        data : Series or DictOfSeries
            The data of the row as a Series if squeeze is True, as
            a DictOfSeries otherwise.

        See Also
        --------
        DictOfSeries.iteritems : Iterate over (column name, Series) pairs.
        """

        # todo: 2nd posibility for fill_value=Any, squeeze=False
        #   do it like in case fill_value=None ->
        #       1. row = aloc the row
        #       2. e = row.isempty()
        #       3. row.loc[idx,e] = fill_value
        #   This approach could be much better, because the dtype of
        #   the columns is preserved.

        # PROBABLY PERFORMANCE EXPENSIVE
        if fill_value is None:
            allidx = self.index_of("all")
            if squeeze:
                for i in allidx:
                    yield i, self.aloc[i:i].dropempty().squeeze(axis=0)
            else:
                for i in allidx:
                    yield self.aloc[i:i]

        # PROBABLY MEMORY EXPENSIVE
        else:
            if fill_value is np.nan:
                df = self.to_df()
            else:
                nans = self.isna().to_df().fillna(False)
                df = self.to_df().fillna(fill_value)
                df[nans] = np.nan
            if squeeze:
                yield from df.iterrows()
            else:
                for idx, row in df.iterrows():
                    yield idx, self._constructor(
                        data=row.to_dict(), index=[idx]
                    )._finalize(self)

    # ------------------------------------------------------------------------------
    # Broadcasting and Reducing

    def for_each(self, attr_or_callable, **kwds):
        """
        Apply a callable or a pandas.Series method or property on each column.

        Parameters
        ----------
        attr_or_callable: Any
            A pandas.Series attribute or any callable, to apply on each column.
            A series attribute can be any property, field or method and also
            could be specified as string. If a callable is given it must take
            pandas.Series as the only positional argument and return a scalar.

        **kwds: any
            kwargs to passed to callable

        Returns
        -------
        pandas.Series
            A series with the results, indexed by the column labels.

        Notes
        -----
        The called function or the attribute works on the actual underlying series.
        If the provided function works inplace it can and will modify the actual data.
        If this is not desired one can should make an explicit copy beforehand. If the
        function returns new objects or copies, explicit copying is not needed.

        See Also
        --------
        DictOfSeries.apply : Apply functions to columns and convert
                             result to DictOfSeries.

        Examples
        --------
        >>> d = DictOfSeries([range(3), range(4)], columns=['a', 'b'])
        >>> d
           a |    b |
        ==== | ==== |
        0  0 | 0  0 |
        1  1 | 1  1 |
        2  2 | 2  2 |
             | 3  3 |

        Use with a callable..

        >>> d.for_each(max)
        columns
        a    2
        b    3
        dtype: object

        ..or with a string, denoting a pd.Series attribute and
        therefor is the same as giving the latter.

        >>> d.for_each('max')
        columns
        a    2
        b    3
        dtype: object

        >>> d.for_each(pd.Series.max)
        columns
        a    2
        b    3
        dtype: object

        Both also works with properties:

        >>> d.for_each('dtype')
        columns
        a    int64
        b    int64
        dtype: object
        """
        attrOcall = attr_or_callable
        if isinstance(attrOcall, str):
            attrOcall = getattr(pd.Series, attrOcall)
        call = callable(attrOcall)
        if not call:
            attrOcall = attr_or_callable
        data = pd.Series(dtype="O", index=self.columns)
        for c, series in self.items():
            if call:
                data.at[c] = attrOcall(series, **kwds)
            else:
                data.at[c] = getattr(series, attrOcall)
        return data

    def apply(self, func, axis=0, raw=False, args=(), **kwds):
        """
        Apply a function along an axis of the DictOfSeries.

        Parameters
        ----------
        func : callable
            Function to apply on each column.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:

            * 0 or 'index': apply function to each column.
            * 1 or 'columns': NOT IMPLEMENTED

        raw : bool, default False
            Determines if row or column is passed as a Series or ndarray object:

            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray objects
              instead.
              If you are just applying a NumPy reduction function this will
              achieve much better performance.

        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        **kwds
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` along the given axis of the
            DataFrame.

        Notes
        -----
        The called function or the attribute works on the actual underlying series.
        If the provided function works inplace it can and will modify the actual data.
        If this is not desired one should make an explicit copy beforehand. If the
        function returns new objects or copies, and does not mess with the data, explicit
        copying is not needed.


        Raises
        ------
        NotImplementedError
            * if axis is 'columns' or 1

        See Also
        --------
        DictOfSeries.for_each: apply pd.Series methods or properties to each column

        Examples
        --------

        We use the example DictOfSeries from :ref:`indexing <doc_indexing:Example dios>`.

        >>> di = di[:5]
            a |    b |     c |     d |
        ===== | ==== | ===== | ===== |
        0   0 | 2  5 | 4   7 | 6   0 |
        1   7 | 3  6 | 5  17 | 7   1 |
        2  14 | 4  7 | 6  27 | 8   2 |
        3  21 | 5  8 | 7  37 | 9   3 |
        4  28 | 6  9 | 8  47 | 10  4 |

        >>> di.apply(max)
        columns
        a    28
        b     9
        c    47
        d     4
        dtype: int64

        >>> di.apply(pd.Series.count)
        columns
        a    5
        b    5
        c    5
        d    5
        dtype: int64

        One can pass keyword arguments directly..

        >>> di.apply(pd.Series.value_counts, normalize=True)
              a |      b |       c |      d |
        ======= | ====== | ======= | ====== |
        7   0.2 | 7  0.2 | 7   0.2 | 4  0.2 |
        14  0.2 | 6  0.2 | 37  0.2 | 3  0.2 |
        21  0.2 | 5  0.2 | 47  0.2 | 2  0.2 |
        28  0.2 | 9  0.2 | 27  0.2 | 1  0.2 |
        0   0.2 | 8  0.2 | 17  0.2 | 0  0.2 |

        Or define a own funtion..

        >>> di.apply(lambda s : 'high' if max(s) > 10 else 'low')
        columns
        a    high
        b     low
        c    high
        d     low
        dtype: object

        And also more advanced functions that return a list-like can be given. Note that
        the returned lists not necessarily must have the same length.

        >>> func = lambda s : ('high', max(s), min(s)) if min(s) > (max(s)//2) else ('low',max(s))
        >>> di.apply(func)
             a |       b |      c |      d |
        ====== | ======= | ====== | ====== |
        0  low | 0  high | 0  low | 0  low |
        1   28 | 1     9 | 1   47 | 1    4 |
               | 2     5 |        |        |
        """
        if axis in [1, "columns"]:
            raise NotImplementedError

        if axis not in [0, "index"]:
            raise ValueError(axis)

        # we cannot use self._data.apply(func=func, args=args, **kwds)
        # because this may return a pandas.DataFrame. Also we cannot
        # use pandas.Series.apply(), because this works on its values.
        need_dios = need_convert = False
        result = pd.Series(dtype="O", index=self.columns)
        for c, series in self.items():
            series = series.values if raw else series
            s = func(series, *args, **kwds)
            result.at[c] = s
            if pdextra.is_scalar(s):
                need_convert = True
            else:
                need_dios = True
                if not isinstance(s, pd.Series):
                    need_convert = True
        if need_dios:
            if need_convert:
                for c, val in result.items():
                    result.at[c] = pd.Series(val)
            itype = _find_least_common_itype(result)
            result = self._constructor(data=result, itype=itype, fastpath=True)
            result._finalize(self)

        return result

    def reduce_columns(self, func, initial=None, skipna=False):
        """
        Reduce all columns to a single pandas.Series by a given function.

        Apply a function of two pandas.Series as arguments, cumulatively to all
        columns, from left to right, so as to reduce the columns to a single
        pandas.Series. If initial is present, it is placed before the columns
        in the calculation, and serves as a default when the columns are empty.

        Parameters
        ----------
        func : function
            The function must take two identically indexed pandas.Series and should
            return a single pandas.Series with the same index.

        initial : column-label or pd.Series, default None
            The series to start with. If None a dummy series is created, with the
            indices of all columns and the first seen values.

        skipna : bool, default False
               If True, skip NaN values.

        Returns
        -------
        pandas.Series
            A series with the reducing result and the index of the start series,
            defined by ``initializer``.
        """
        if initial is None:
            value = pd.Series(index=self.index_of("all"))
            for d in self._data:
                value = value.combine_first(d)
        elif isinstance(initial, pd.Series):
            value = initial.copy()
        elif initial in self.columns:
            value = self._data.at[initial].copy()
        else:
            raise ValueError("initial must be pd.Series, a column label or None")

        if skipna:
            val = value.dropna()
            data = self.dropna()._data
        else:
            val = value
            data = self._data

        for d in data:
            idx = val.index & d.index
            if len(idx) > 0:
                l, r = val.loc[idx], d.loc[idx]
                val.loc[idx] = func(l, r)

        if skipna:
            value.loc[val.index] = val
        return value

    # ------------------------------------------------------------------------------
    # Merging and Joining

    def combine_first(self, other, keepna=False):
        """
        Update null elements with value in the same location in other.

        Combine two DictOfSeries objects by filling null values in one DictOfSeries with
        non-null values from other DictOfSeries. The row and column indexes of the resulting
        DictOfSeries will be the union of the two.

        Parameters
        ----------
        keepna : bool, default False
            By default Nan's are updated by other and new value-index pairs from other are
            inserted. If set to True, NaN's are not updated and only new value-index pair are inserted.

        other : DictOfSeries
            Provided DictOfSeries to use to fill null values.

        Returns
        -------
        DictOfSeries
        """
        if keepna:
            nans = self.isna()

        new: DictOfSeries = self.copy()
        for c in other.columns:
            if c in self.columns:
                col = self._data.at[c].combine_first(other[c])
            else:
                col = other[c]
            new._data.at[c] = col

        if keepna:
            new.aloc[nans] = np.nan

        return new

    # ------------------------------------------------------------------------------
    # Misc methods

    def index_of(self, method="all"):
        """Return an single index with indices from all columns.

        Parameters
        ----------
        method : string, default 'all'
            * 'all' : get all indices from all columns
            * 'union' : alias for 'all'
            * 'shared' : get indices that are present in every columns
            * 'intersection' : alias for 'shared'
            * 'uniques' : get indices that are only present in a single column
            * 'non-uniques' : get indices that are present in more than one column

        Returns
        -------
        pd.Index
            A single duplicate-free index, somehow representing indices of all columns.

        Examples
        --------
        We use the example DictOfSeries from :ref:`indexing <doc_indexing:Example dios>`.

        >>> di
            a |      b |      c |     d |
        ===== | ====== | ====== | ===== |
        0   0 | 2    5 | 4    7 | 6   0 |
        1   7 | 3    6 | 5   17 | 7   1 |
        2  14 | 4    7 | 6   27 | 8   2 |
        3  21 | 5    8 | 7   37 | 9   3 |
        4  28 | 6    9 | 8   47 | 10  4 |
        5  35 | 7   10 | 9   57 | 11  5 |
        6  42 | 8   11 | 10  67 | 12  6 |
        7  49 | 9   12 | 11  77 | 13  7 |
        8  56 | 10  13 | 12  87 | 14  8 |
        9  63 | 11  14 | 13  97 | 15  9 |

        >>> di.index_of()
        RangeIndex(start=0, stop=16, step=1)

        >>> di.index_of("shared")
        Int64Index([6, 7, 8, 9], dtype='int64')

        >>> di.index_of("uniques")
        Int64Index([0, 1, 14, 15], dtype='int64')
        """
        indexes = self.indexes
        if len(indexes) <= 1:
            return indexes.squeeze()

        if method in ["union", "all"]:
            res = ftools.reduce(pd.Index.union, indexes)
        elif method in ["intersection", "shared"]:
            res = ftools.reduce(pd.Index.intersection, indexes)
        elif method in ["uniques", "non-uniques"]:
            res = ftools.reduce(pd.Index.append, indexes)
            res = res.value_counts(sort=False, dropna=False)
            if method == "uniques":
                res = res[res == 1].index
            else:
                res = res[res > 1].index
        else:
            raise ValueError(method)
        return res if res.is_unique else res.unique()

    def squeeze(self, axis=None):
        """Squeeze a 1-dimensional axis objects into scalars."""
        if axis in [0, "index"]:
            if (self.lengths == 1).all():
                return self._data.apply(pd.Series.squeeze)
            return self
        elif axis in [1, "columns"]:
            if len(self) == 1:
                return self._data.squeeze()
            return self
        elif axis is None:
            if len(self) == 1:
                return self._data.squeeze().squeeze()
            if (self.lengths == 1).all():
                return self._data.apply(pd.Series.squeeze).squeeze()
            return self
        raise ValueError(axis)

    def dropna(self, inplace=False):
        """Return a bolean array that is `True` if the value is a Nan-value"""
        data = self.for_each("dropna", inplace=inplace)
        if inplace:
            return
        return self._construct_like_self(data=data, fastpath=True)

    def dropempty(self):
        """Drop empty columns. Return copy."""
        return self.loc[:, self.notempty()]

    def astype(self, dtype, copy=True, errors="raise"):
        """Cast the data to the given data type."""
        data = self.for_each("astype", dtype=dtype, copy=copy, errors=errors)
        return self._construct_like_self(data=data, fastpath=True)

    def _mask_or_where(self, cond, other=np.nan, inplace=False, mask=True):
        """helper to mask/where"""
        data = self if inplace else self.copy()

        if callable(other):
            other = other(data)

        if callable(cond):
            cond = cond(data)
        # if DictOfSeries is bool,
        # is already checked in aloc
        elif not _is_dios_like(cond):
            if not pdextra.is_bool_indexer(cond):
                raise ValueError(
                    "Object with boolean values only expected as condition"
                )

        if mask:
            data.aloc[cond] = other
        else:
            data.aloc[~cond] = other

        if inplace:
            return None
        return data

    def where(self, cond, other=np.nan, inplace=False):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool DictOfSeries, Series, array-like, or callable
            Where cond is True, keep the original value. Where False, replace
            with corresponding value from other. If cond is callable, it is computed
            on the DictOfSeries and should return boolean DictOfSeries or array.
            The callable must not change input DictOfSeries (though dios doesn’t check it).
            If cond is a bool Series, every column is (row-)aligned against it, before the
            boolean values are evaluated. Missing indices are treated like False values.

        other : scalar, Series, DictOfSeries, or callable
            Entries where cond is False are replaced with corresponding value from other.
            If other is callable, it is computed on the DictOfSeries and should return scalar
            or DictOfSeries. The callable must not change input DictOfSeries (though dios doesn’t check it).
            If other is a Series, every column is (row-)aligned against it, before the values
            are written. NAN's are written for missing indices.

        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        DictOfSeries

        See Also
        --------
        mask: Mask data where condition is True
        """
        return self._mask_or_where(cond=cond, other=other, inplace=inplace, mask=False)

    def mask(self, cond, other=np.nan, inplace=False):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool DictOfSeries, Series, array-like, or callable
            Where cond is False, keep the original value. Where True, replace
            with corresponding value from other. If cond is callable, it is computed
            on the DictOfSeries and should return boolean DictOfSeries or array.
            The callable must not change input DictOfSeries (though dios doesn’t check it).
            If cond is a bool Series, every column is (row-)aligned against it, before the
            boolean values are evaluated. Missing indices are treated like False values.

        other : scalar, Series, DictOfSeries, or callable
            Entries where cond is True are replaced with corresponding value from other.
            If other is callable, it is computed on the DictOfSeries and should return scalar
            or DictOfSeries. The callable must not change input DictOfSeries (though dios doesn’t check it).
            If other is a Series, every column is (row-)aligned against it, before the values
            are written. NAN's are written for missing indices.

        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        DictOfSeries

        See Also
        --------
        mask: Mask data where condition is False
        """
        return self._mask_or_where(cond=cond, other=other, inplace=inplace, mask=True)

    def memory_usage(self, index=True, deep=False):
        return self.for_each(pd.Series.memory_usage, index=index, deep=deep).sum()

    def to_df(self, how="outer"):
        """
        Transform DictOfSeries to a pandas.DataFrame.

        Because a pandas.DataFrame can not handle Series of different
        length, but DictOfSeries can, the missing data is filled with
        NaNs or is dropped, depending on the keyword `how`.

        Parameters
        ----------
        how: {'outer', 'inner'}, default 'outer'
            define how the resulting DataFrame index is generated:
            * 'outer': The indices of all columns, merged into one index is used.
                If a column misses values at the new index location, `NaN`s are filled.
            * 'inner': Only indices that are present in all columns are used, filling
                logic is not needed, but values are dropped, if a column has indices
                that are not known to all other columns.

        Returns
        -------
        pandas.DataFrame: transformed data

        Examples
        --------

        Missing data locations are filled with NaN's

        >>> a = pd.Series(11, index=range(2))
        >>> b = pd.Series(22, index=range(3))
        >>> c = pd.Series(33, index=range(1,9,3))
        >>> di = DictOfSeries(dict(a=a, b=b, c=c))
        >>> di
            a |     b |     c |
        ===== | ===== | ===== |
        0  11 | 0  22 | 1  33 |
        1  11 | 1  22 | 4  33 |
              | 2  22 | 7  33 |

        >>> di.to_df()
        columns     a     b     c
        0        11.0  22.0   NaN
        1        11.0  22.0  33.0
        2         NaN  22.0   NaN
        4         NaN   NaN  33.0
        7         NaN   NaN  33.0

        or is dropped if `how='inner'`

        >>> di.to_df(how='inner')
        columns   a   b   c
        1        11  22  33
        """
        if how == "inner":
            how = "shared"
        elif how == "outer":
            how = "all"
        else:
            raise ValueError(how)

        index = self.index_of(how)
        df = pd.DataFrame(columns=self.columns, index=index)
        for c, series in self.items():
            # this automatically respects the df-index, that
            # was set before. Missing locations are already
            # nans, present locations are set.
            df[c] = series.copy()

        df.attrs = self.attrs
        return df

    @property
    def debugDf(self):
        """Alias for ``to_df()`` as property, for debugging purpose."""
        return self.to_df()

    def min(self, axis=0, skipna=True):
        if axis is None:
            return self.for_each(pd.Series.min, skipna=skipna).min()
        if axis in [0, "index"]:
            return self.for_each(pd.Series.min, skipna=skipna)
        if axis in [1, "columns"]:
            func = lambda s1, s2: s1.where(s1 < s2, s2)
            return self.reduce_columns(func, skipna=skipna)
        raise ValueError(axis)

    def max(self, axis=0, skipna=None):
        if axis is None:
            return self.for_each(pd.Series.max, skipna=skipna).max()
        if axis in [0, "index"]:
            return self.for_each(pd.Series.max, skipna=skipna)
        if axis in [1, "columns"]:
            func = lambda s1, s2: s1.where(s1 > s2, s2)
            return self.reduce_columns(func, skipna=skipna)
        raise ValueError(axis)

    # ----------------------------------------------------------------------
    # Boolean and empty stuff

    def equals(self, other):
        """
        Test whether two DictOfSeries contain the same elements.

        This function allows two DictOfSeries to be compared against each other to see
        if they have the same shape and elements. NaNs in the same location are considered equal.
        The column headers do not need to have the same type, but the elements within the columns
        must be the same dtype.

        Parameters
        ----------
        other: DictOfSeries
            The other DictOfSeries to compare with.

        Returns
        -------
        bool
            True if all elements are the same in both DictOfSeries, False otherwise.
        """
        if not isinstance(other, _DiosBase):
            return False
        try:
            eq_nans = (self.isna() == other.isna()).all(None)
            eq_data = (self.dropna() == other.dropna()).all(None)
            eq_dtypes = (self.dtypes == other.dtypes).all()
            return eq_nans and eq_dtypes and eq_data
        except Exception:
            return False

    def isin(self, values):
        """Return a boolean dios, that indicates if the corresponding value is in the given array-like."""
        data = self.for_each("isin", values=values)
        return self._construct_like_self(data=data, fastpath=True)

    def all(self, axis=0):
        """
        Return whether all elements are True, potentially over an axis.

        Returns True unless there at least one element within a series
        or along a DictOfSeries axis that is False or equivalent (e.g. zero or empty).

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default 0
            Indicate which axis or axes should be reduced.
             * 0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.
             * 1 / ‘columns’ : reduce the columns, return a Series whose index is the union of all columns indexes.
             * None : reduce all axes, return a scalar.

        Returns
        -------
        pandas.Series

        See Also
        --------
        pandas.Series.all: Return True if all elements are True.
        any: Return True if one (or more) elements are True.
        """
        if axis is None:
            return self._data.apply(all).all()
        if axis in [0, "index"]:
            return self._data.apply(all)
        if axis in [1, "columns"]:
            func = lambda s1, s2: s1.astype(bool) & s2.astype(bool)
            init = pd.Series(True, dtype=bool, index=self.index_of("all"))
            return self.reduce_columns(func, init)
        raise ValueError(axis)

    def any(self, axis=0):
        """
        Return whether any element is True, potentially over an axis.

        Returns False unless there at least one element within a series
        or along a DictOfSeries axis that is True or equivalent (e.g. non-zero or non-empty).

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default 0
            Indicate which axis or axes should be reduced.
             * 0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.
             * 1 / ‘columns’ : reduce the columns, return a Series whose index is the union of all columns indexes.
             * None : reduce all axes, return a scalar.

        Returns
        -------
        pandas.Series

        See Also
        --------
        pandas.Series.any: Return whether any element is True.
        all: Return True if all elements are True.
        """
        if axis is None:
            return self._data.apply(any).any()
        if axis in [0, "index"]:
            return self._data.apply(any)
        if axis in [1, "columns"]:
            func = lambda s1, s2: s1.astype(bool) | s2.astype(bool)
            init = pd.Series(False, dtype=bool, index=self.index_of("all"))
            return self.reduce_columns(func, init)
        raise ValueError(axis)

    def isna(self, drop_empty=False):
        """
        Return a boolean DictOfSeries which indicates NA positions.
        """
        data = self.dropempty() if drop_empty else self
        data = data.for_each("isna")
        return self._construct_like_self(data=data, fastpath=True)

    def notna(self, drop_empty=False):
        """
        Return a boolean DictOfSeries which indicates non-NA positions.
        """
        data = self.dropempty() if drop_empty else self
        data = data.for_each("notna")
        return self._construct_like_self(data=data, fastpath=True)

    def hasnans(self, axis=0, drop_empty=False):
        """
        Returns a boolean Series along an axis, which indicates if it contains NA-entries.
        """
        data = self.dropempty() if drop_empty else self
        if axis is None:
            return data.for_each("hasnans").any()
        if axis in [0, "index"]:
            return data.for_each("hasnans")
        if axis in [1, "columns"]:
            func = lambda s1, s2: s1.isna() | s2.isna()
            init = pd.Series(False, dtype=bool, index=self.index_of("all"))
            return data.reduce_columns(func, init)
        raise ValueError(axis)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        if axis in [None, 0, "index"]:
            kws = dict(value=value, method=method, limit=limit, downcast=downcast)
            data = self.for_each("fillna", inplace=inplace, **kws)
            if inplace:
                return
            return self._construct_like_self(data=data, fastpath=True)

        if axis in [1, "columns"]:
            raise NotImplementedError
        raise ValueError(axis)

    def isempty(self):
        """Returns a boolean Series, which indicates if an column is empty"""
        return self.for_each("empty").astype(bool)

    def notempty(self):
        """Returns a boolean Series, which indicates if an column is not empty"""
        return ~self.isempty()

    def isdata(self):
        """Alias for ``notna(drop_empty=True)``."""
        return self.notna(drop_empty=True)

    def isnull(self, drop_empty=False):
        """Alias for ``isna()``"""
        return self.isna(drop_empty=drop_empty)

    def notnull(self, drop_empty=False):
        """Alias, see ``notna()``."""
        return self.notna(drop_empty=drop_empty)

    def to_dios(self):
        """
        A dummy to allow unconditional to_dios calls
        on pd.DataFrame, pd.Series and dios.DictOfSeries
        """
        return self

    # ----------------------------------------------------------------------
    # Rendering Methods

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr = dios_options[OptsFields.dios_repr]
        showdim = self.lengths.max() > dios_options[OptsFields.disp_max_rows]
        return self.to_string(method=repr, show_dimensions=showdim)

    def to_string(
        self,
        max_rows=None,
        min_rows=None,
        max_cols=None,
        na_rep="NaN",
        show_dimensions=False,
        method=Opts.repr_indexed,
        no_value=" ",
        empty_series_rep="no data",
        col_delim=" | ",
        header_delim="=",
        col_space=None,
    ):
        """Pretty print a dios.

        if `method` == `indexed` (default):
            every column is represented by a own index and corresponding values

        if `method` == `aligned` [2]:
            one(!) global index is generated and values from a column appear at
            the corresponding index-location.

        Parameters
        ---------

        max_cols :
            not more column than `max_cols` are printed [1]

        max_rows :
            see `min_rows` [1]

        min_rows :
            not more rows than `min_rows` are printed, if rows of any series exceed `max_rows` [1]

        na_rep :
            all NaN-values are replaced by `na_rep`. Default `NaN`

        empty_series_rep :
            Ignored if not `method='indexed'`.
            Empty series are represented by the string in `empty_series_rep`

        col_delim : str
            Ignored if not `method='indexed'`.
            between all columns `col_delim` is inserted.

        header_delim :
            Ignored if not `method='indexed'`.
            between the column names (header) and the data, `header_delim` is inserted,
            if not None. The string is repeated, up to the width of the column. (str or None).

        no_value :
            Ignored if not `method='aligned'`.
            value that indicates, that no entry in the underling series is present. Bear in mind
            that this should differ from `na_rep`, otherwise you cannot differ missing- from NaN- values.

        Notes
        -----
            [1]: defaults to the corresponding value in `dios_options`
            [2]: the common-params are directly passed to pd.DataFrame.to_string(..)
            under the hood, if method is `aligned`

        """
        if self.empty:
            return _empty_repr(self)

        max_cols = max_cols or dios_options[OptsFields.disp_max_cols] or 100
        max_rows = max_rows or dios_options[OptsFields.disp_max_rows] or 200
        min_rows = min_rows or dios_options[OptsFields.disp_min_rows] or 100

        kwargs = dict(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            na_rep=na_rep,
            col_space=col_space,
            show_dimensions=show_dimensions,
        )

        if method == Opts.repr_aligned:
            return _to_aligned_df(self, no_value=no_value).to_string(**kwargs)

        # add pprint relevant options
        kwargs.update(
            empty_series_rep=empty_series_rep,
            col_delim=col_delim,
            header_delim=header_delim,
        )

        return pprint_dios(self, **kwargs)

    def to_csv(self, *args, **kwargs):
        self.to_df().to_csv(*args, **kwargs)

    to_csv.__doc__ = pd.DataFrame.to_csv.__doc__


def _empty_repr(di):
    return f"Empty DictOfSeries\n" f"Columns: {di.columns.to_list()}"


def pprint_dios(
    dios,
    max_rows=None,
    min_rows=None,
    max_cols=None,
    na_rep="NaN",
    empty_series_rep="no data",
    col_space=None,
    show_dimensions=True,
    col_delim=" | ",
    header_delim="=",
):
    na_rep = str(na_rep)
    empty_series_rep = str(empty_series_rep)
    col_delim = col_delim or " "

    min_rows = min(max_rows, min_rows)

    if dios.empty:
        return _empty_repr(dios)

    maxlen = dios.lengths.max()
    data = dios._data

    trunc_cols = len(data) > max_cols
    if trunc_cols:
        left, right = data.head(max_cols // 2), data.tail(max_cols // 2)
        data = left.append(right)

    # now data only contains series that we want to print.

    # if any series exceed max_rows we trim all series to min_rows
    series_lengths = data.apply(len).to_list()
    series_maxlen = max(series_lengths)
    trunc_rows = series_maxlen > max_rows
    max_rows = min_rows if trunc_rows else series_maxlen

    # we make a list of list, where the inner contains all
    # stringified values of the series upto max_rows+1, where
    # the additional row is the column-name
    outer = []
    for colname in data.index:
        s = data.at[colname]

        isempty = s.empty
        if isempty:
            s = pd.Series(empty_series_rep)
            idx = False
            cspace = col_space
        else:
            idx = True
            cspace = col_space // 2 if col_space else col_space

        sstr = s.to_frame().to_string(
            col_space=cspace,
            header=[str(colname)],
            index=idx,
            na_rep=na_rep,
            max_rows=max_rows,
            min_rows=min_rows,
        )
        li = sstr.split("\n")

        # HACK: empty series produce a unnecessary space,
        # because index is omitted
        if isempty:
            cstr, vstr = li
            if len(cstr.lstrip()) < len(vstr) and (cspace or 0) < len(vstr):
                li = [cstr[1:], vstr[1:]]

        outer.append(li)

    # now the length of every value-string per series are the same.
    # we need this length's to know, how many chars we need to fill,
    # once we exceed the length of the series, or if we insert whole
    # columns.
    valstr_len = [len(c[0]) for c in outer]

    rows = max_rows + 1  # colnames aka. header
    rows += 1 if trunc_rows else 0  # `...` in rows
    rows += 1 if header_delim else 0  # underline header

    if header_delim:
        for i, c in enumerate(outer):
            colheader = (header_delim * valstr_len[i])[: valstr_len[i]]
            c.insert(1, colheader)

    dots = " ... "
    if trunc_cols:
        outer.insert(max_cols // 2, [dots] * rows)
        valstr_len.insert(max_cols // 2, len(dots))
        series_lengths.insert(max_cols // 2, rows)

    txt = ""
    for r in range(rows):
        for i, c in enumerate(outer):
            try:
                vstr = c[r]
            except IndexError:
                vstr = " " * valstr_len[i]
            txt += vstr + col_delim
        txt += "\n"

    # add footer
    if show_dimensions:
        for i, c in enumerate(outer):
            # ignore the dot-column
            if trunc_cols and i == max_cols // 2:
                txt += dots + " " * len(col_delim)
            else:
                txt += f"[{series_lengths[i]}]".ljust(valstr_len[i] + len(col_delim))

        txt += f"\n\nmax: [{maxlen} rows x {len(dios.columns)} columns]"
        txt += "\n"

    return txt


def _to_aligned_df(dios, no_value=" "):
    if dios.empty:
        return pd.DataFrame(columns=dios.columns)

    # keep track of all real nans
    nandict = {}
    for c in dios:
        nans = dios[c].isna()
        nandict[c] = nans[nans].index

    df = dios.to_df()
    df[df.isna()] = no_value

    # reinsert all real nans
    for c in df:
        df.loc[nandict[c], c] = np.nan

    return df


def to_dios(obj) -> DictOfSeries:
    if isinstance(obj, DictOfSeries):
        return obj
    return DictOfSeries(data=obj)


def __monkey_patch_pandas():
    def to_dios(self):
        return DictOfSeries(data=self)

    pd.Series.to_dios = to_dios
    pd.DataFrame.to_dios = to_dios


__monkey_patch_pandas()
