from . import pandas_bridge as pdextra
from .base import (
    _DiosBase,
    _is_dios_like,
    _is_bool_dios_like,
)

import numpy as np
import pandas as pd


class _Indexer:
    def __init__(self, obj: _DiosBase):
        self.obj = obj
        self._data = obj._data

    def _unpack_key(self, key):

        key = list(key) if pdextra.is_iterator(key) else key

        if isinstance(key, tuple):
            if len(key) > 2:
                raise KeyError("To many indexers")
            rowkey, colkey = key
        else:
            rowkey, colkey = key, slice(None)

        if isinstance(rowkey, tuple) or isinstance(colkey, tuple):
            raise KeyError(f"{key}. tuples are not allowed.")

        rowkey = list(rowkey) if pdextra.is_iterator(rowkey) else rowkey
        colkey = list(colkey) if pdextra.is_iterator(colkey) else colkey
        return rowkey, colkey

    def _set_value_muli_column(self, rowkey, colkey, value, xloc="loc"):
        """set value helper for loc and iloc"""

        data = getattr(self._data, xloc)[colkey]

        hashable_rkey = pdextra.is_hashable(rowkey)
        dioslike_value = False
        iter_value = False

        if _is_dios_like(value):
            dioslike_value = True
            if hashable_rkey:
                raise ValueError(f"Incompatible indexer with DictOfSeries")

        elif pdextra.is_list_like(value):
            value = value.values if isinstance(value, pd.Series) else value
            iter_value = True
            if len(value) != len(data):
                raise ValueError(
                    f"shape mismatch: value array of shape (.., {len(value)}) could "
                    f"not be broadcast to indexing result of shape (.., {len(data)})"
                )
        c = "?"
        try:
            for i, c in enumerate(data.index):
                dat = data.at[c]
                dat_xloc = getattr(dat, xloc)

                if dioslike_value:
                    # set to empty series fail; emptySer.loc[:] = [2,1]
                    # len(scalar) -> would fail, but cannot happen,
                    # because dioslike+hashable, already was checked
                    if len(dat_xloc[rowkey]) == 0:
                        continue

                # unpack the value if necessary
                if iter_value:
                    val = value[i]
                elif dioslike_value:
                    val = value[c] if c in value else np.nan
                else:
                    val = value

                dat_xloc[rowkey] = val

        except Exception as e:
            raise type(e)(f"failed for column {c}: " + str(e)) from e


# #############################################################################


class _LocIndexer(_Indexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):

        rowkey, colkey = self._unpack_key(key)
        if _is_dios_like(rowkey) or _is_dios_like(colkey):
            raise ValueError("Could not index with multidimensional key")

        # simple optimisation
        if pdextra.is_null_slice(rowkey) and pdextra.is_null_slice(colkey):
            return self.obj.copy()

        data = self._data.loc[colkey].copy()

        # .loc[any, scalar] -> (a single) series
        # .loc[scalar, scalar] -> (a single) value
        if pdextra.is_hashable(colkey):
            new = data.loc[rowkey]

        # .loc[any, non-scalar]
        else:
            k = "?"
            try:

                for k in data.index:
                    data.at[k] = data.at[k].loc[rowkey]

            except Exception as e:
                raise type(e)(f"failed for column {k}: " + str(e)) from e

            # .loc[scalar, non-scalar] -> column-indexed series
            if pdextra.is_hashable(rowkey):
                new = data

            # .loc[non-scalar, non-scalar] -> dios
            else:
                new = self.obj.copy_empty(columns=False)
                new._data = data

        return new

    def __setitem__(self, key, value):

        rowkey, colkey = self._unpack_key(key)
        if _is_dios_like(rowkey) or _is_dios_like(colkey):
            raise ValueError("Cannot index with multi-dimensional key")

        # .loc[any, scalar] - set on single column
        if pdextra.is_hashable(colkey):

            # .loc[dont-care, new-scalar] = val
            if colkey not in self.obj.columns:
                self.obj._insert(colkey, value)

            # .loc[any, scalar] = multi-dim
            elif _is_dios_like(value) or pdextra.is_nested_list_like(value):
                raise ValueError("Incompatible indexer with multi-dimensional value")

            # .loc[any, scalar] = val
            else:
                self._data.at[colkey].loc[rowkey] = value

        # .loc[any, non-scalar] = any
        else:
            self._set_value_muli_column(rowkey, colkey, value, xloc="loc")


# #############################################################################


class _iLocIndexer(_Indexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        rowkey, colkey = self._unpack_key(key)
        if _is_dios_like(rowkey) or _is_dios_like(colkey):
            raise ValueError("Cannot index with multidimensional key")

        # simple optimisation
        if pdextra.is_null_slice(rowkey) and pdextra.is_null_slice(colkey):
            return self.obj.copy()

        data = self._data.iloc[colkey].copy()

        # .iloc[any, int] -> single series
        # .iloc[int, int] -> single value
        if pdextra.is_integer(colkey):
            new = data.iloc[rowkey]

        # .iloc[any, non-int]
        else:
            k = "?"
            try:

                for k in data.index:
                    data.at[k] = data.at[k].iloc[rowkey]

            except Exception as e:
                raise type(e)(f"failed for column {k}: " + str(e)) from e

            # .iloc[int, non-int] -> column-indexed series
            if pdextra.is_integer(rowkey):
                new = data

            # .iloc[non-int, non-int] -> dios
            else:
                new = self.obj.copy_empty(columns=False)
                new._data = data

        return new

    def __setitem__(self, key, value):
        rowkey, colkey = self._unpack_key(key)
        if _is_dios_like(rowkey) or _is_dios_like(colkey):
            raise ValueError("Cannot index with multidimensional key")

        # .iloc[any, int] = Any
        if pdextra.is_integer(colkey):
            if _is_dios_like(value) or pdextra.is_nested_list_like(value):
                raise ValueError("Incompatible indexer with multi-dimensional value")
            self._data.iat[colkey].iloc[rowkey] = value

        # .iloc[any, non-int] = Any
        else:
            self._set_value_muli_column(rowkey, colkey, value, xloc="iloc")


# #############################################################################


class _aLocIndexer(_Indexer):
    """align Indexer

    Automatically align (alignable) indexer on all possible axis,
    and handle indexing with non-existent or missing keys gracefully.

    Also align (alignable) values before setting them with .loc
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._usebool = True

    def __call__(self, usebool=True):
        """We are called if the user want to set `usebool=False', which make
        boolean alignable indexer treat as non-boolean alignable indexer.

        Explanation: A boolean dios indexer align its indices with the indices
        of the receiving dios like a non-boolean dios indexer also would do.
        Additionally all rows with False values are kicked too. To disable
         that `usebool=False` can be given."""
        self._usebool = usebool
        return self

    def __getitem__(self, key):
        rowkeys, colkeys, lowdim = self._unpack_key_aloc(key)
        data = pd.Series(dtype="O", index=colkeys)
        kws = dict(itype=self.obj.itype, cast_policy=self.obj._policy)

        c = "?"
        try:

            for i, c in enumerate(data.index):
                data.at[c] = self._data.at[c].loc[rowkeys[i]]

        except Exception as e:
            raise type(e)(f"failed for column {c}: " + str(e)) from e

        if lowdim:
            return data.squeeze()
        else:
            return self.obj._constructor(data=data, fastpath=True, **kws)._finalize(
                self.obj
            )

    def __setitem__(self, key, value):
        rowkeys, colkeys, _ = self._unpack_key_aloc(key)

        def iter_self(colkeys, position=False):
            c = "?"
            try:

                for i, c in enumerate(colkeys):
                    dat = self._data.at[c]
                    rk = rowkeys[i]
                    if len(dat.loc[rk]) == 0:
                        continue
                    yield dat, rk, i if position else c

            except Exception as e:
                raise type(e)(f"failed for column {c}: " + str(e)) from e

        # align columns, for rows use series.loc to align
        if _is_dios_like(value):
            colkeys = value.columns.intersection(colkeys)
            for dat, rk, c in iter_self(colkeys):
                dat.loc[rk] = value[c]

        # no align, no merci
        elif pdextra.is_nested_list_like(value):
            if len(colkeys) != len(value):
                raise ValueError(
                    f"shape mismatch: values array of shape "
                    f"(.., {len(value)}) could not "
                    f"be broadcast to indexing result of "
                    f"shape (.., {len(colkeys)})"
                )
            for dat, rk, i in iter_self(colkeys, position=True):
                dat.loc[rk] = value[i]

        # align rows by using series.loc
        elif isinstance(value, pd.Series):
            for dat, rk, _ in iter_self(colkeys):
                dat.loc[rk] = value

        # no align, no merci
        else:
            for dat, rk, _ in iter_self(colkeys):
                dat.loc[rk] = value

    def _unpack_key_aloc(self, key):
        """
        Return a list of row indexer and a list of existing(!) column labels.
        Both list always have the same length and also could be empty together.

        Note:
            The items of the row indexer list should be passed to pd.Series.loc[]
        """
        # if a single column-key is given, the caller may
        # want to return a single Series, instead of a dios
        lowdim = False

        def keys_from_bool_dios_like(key):
            if not _is_bool_dios_like(key):
                raise ValueError("Must pass dios-like key with boolean values only.")
            colkey = self.obj.columns.intersection(key.columns)
            rowkey = []
            for c in colkey:
                b = key[c]
                rowkey += [self._data.at[c].index.intersection(b[b].index)]
            return rowkey, colkey, lowdim

        def keys_from_dios_like(key):
            colkey = self.obj.columns.intersection(key.columns)
            rowkey = [self._data.at[c].index.intersection(key[c].index) for c in colkey]
            return rowkey, colkey, lowdim

        rowkey, colkey = self._unpack_key(key)

        if _is_dios_like(colkey) or pdextra.is_nested_list_like(colkey):
            raise ValueError("Could not index with multi-dimensional column key.")

        # giving the ellipsis as column key, is an alias
        # for giving `usebool=False`. see self.__call__()
        if colkey is Ellipsis:
            self._usebool = False
            colkey = slice(None)

        # .aloc[dios]
        if _is_dios_like(rowkey):

            if not pdextra.is_null_slice(colkey):
                raise ValueError(
                    f"Could not index with a dios-like indexer as rowkey,"
                    f"and a column key of that type {type(colkey)}"
                )
            if self._usebool:
                return keys_from_bool_dios_like(rowkey)
            else:
                return keys_from_dios_like(rowkey)

        # handle gracefully: scalar
        elif pdextra.is_hashable(colkey):
            colkey = [colkey] if colkey in self.obj.columns else []
            lowdim = True

        # column-alignable: list-like, filter only existing columns
        elif pdextra.is_list_like(colkey) and not pdextra.is_bool_indexer(colkey):
            colkey = colkey.values if isinstance(colkey, pd.Series) else colkey
            colkey = self.obj.columns.intersection(colkey)

        # handle gracefully (automatically)
        # just a simple optimisation
        elif pdextra.is_null_slice(colkey):
            colkey = self.obj.columns

        # not alignable, fall back to .loc (boolean list/series, slice(..), etc.
        else:
            colkey = self._data.loc[colkey].index

        if len(colkey) == 0:  # (!) `if not colkey:` fails for pd.Index
            return [], [], lowdim

        rowkey = self._get_rowkey(rowkey, colkey)

        return rowkey, colkey, lowdim

    def _get_rowkey(self, rowkey, colkey, depth=0):

        if pdextra.is_nested_list_like(rowkey) and depth == 0:
            rowkey = rowkey.values if isinstance(rowkey, pd.Series) else rowkey
            if len(rowkey) != len(colkey):
                raise ValueError(
                    "Nested arrays indexer must have same (outer) "
                    "length than the number of selected columns."
                )
            indexer = []
            for i, c in enumerate(colkey):
                # recurse to get the row indexer from inner element
                indexer += self._get_rowkey(rowkey[i], [c], depth=depth + 1)
            rowkey = indexer

        # row-alignable: pd.Series(), align rows to every series in colkey (columns)
        elif isinstance(rowkey, pd.Series):
            if self._usebool and pdextra.is_bool_indexer(rowkey):
                rowkey = [
                    self._data.at[c].index.intersection(rowkey[rowkey].index)
                    for c in colkey
                ]
            else:
                rowkey = [
                    self._data.at[c].index.intersection(rowkey.index) for c in colkey
                ]

        # handle gracefully: scalar, transform to row-slice
        elif pdextra.is_hashable(rowkey):
            rowkey = [slice(rowkey, rowkey)] * len(colkey)

        # handle gracefully: list-like, filter only existing rows
        # NOTE: dios.aloc[series.index] is processed here
        elif pdextra.is_list_like(rowkey) and not pdextra.is_bool_indexer(rowkey):
            rowkey = [self._data.at[c].index.intersection(rowkey) for c in colkey]

        # not alignable
        # the rowkey is processed by .loc someway in
        # the calling function - (eg. slice(..), boolean list-like, etc.)
        else:
            rowkey = [rowkey] * len(colkey)

        return rowkey


# #############################################################################


class _AtIndexer(_Indexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_key(self, key):
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and pdextra.is_hashable(key[0])
            and pdextra.is_hashable(key[1])
        ):
            raise KeyError(
                f"{key}. `.at` takes exactly one scalar row-key "
                "and one scalar column-key"
            )

    def __getitem__(self, key):
        self._check_key(key)
        return self._data.at[key[1]].at[key[0]]

    def __setitem__(self, key, value):
        self._check_key(key)
        if _is_dios_like(value) or pdextra.is_nested_list_like(value):
            raise TypeError(
                ".at[] cannot be used to set multi-dimensional values, use .aloc[] instead."
            )
        self._data.at[key[1]].at[key[0]] = value


# #############################################################################


class _iAtIndexer(_Indexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_key(self, key):
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and pdextra.is_integer(key[0])
            and pdextra.is_integer(key[1])
        ):
            raise KeyError(
                f"{key} `.iat` takes exactly one integer positional "
                f"row-key and one integer positional scalar column-key"
            )

    def __getitem__(self, key):
        self._check_key(key)
        return self._data.iat[key[1]].iat[key[0]]

    def __setitem__(self, key, value):
        self._check_key(key)
        if _is_dios_like(value) or pdextra.is_nested_list_like(value):
            raise TypeError(
                ".iat[] cannot be used to set multi-dimensional values, use .aloc[] instead."
            )
        self._data.iat[key[1]].iat[key[0]] = value
