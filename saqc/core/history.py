#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy as _copy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from saqc import UNFLAGGED

AGGRGEGATIONS = {
    "last": lambda x: x.ffill(axis=1).iloc[:, -1],
    "max": lambda x: x.max(axis=1),
    "min": lambda x: x.min(axis=1),
}
AGGREGATION = "last"


class History:
    """
    Saqc internal storage for the history of a (single) flags column.

    The flag-history (FH) stores the history of a flags column. Each time
    ``append`` is called a new column is appended to the FH. The column
    names are increasing integers starting with 0. After initialisation
    the FH is empty and has no columns at all.

    To get the latest flags, that are currently stored in the FH, we provide
    a ``squeeze()`` method.

    For more details and a detailed discussion, why this is needed, how this
    works and possible other implementations, see #GL143 [1].

    [1] https://git.ufz.de/rdm-software/saqc/-/issues/143

    Parameters
    ----------
    index: pd.Index
        A index that fit the flags to be insert.

    See also
    --------
    createHistoryFromData: function to create History from existing data
    """

    def __init__(self, index: pd.Index | None):
        self._hist = pd.DataFrame(index=index)
        self._meta = []

    def __getitem__(self, key) -> History:
        if not isinstance(key, tuple):
            # we got a single indexer like hist[3:-4]
            key = (key, slice(None))
        rows, cols = key
        out = History(index=None)
        out._hist = self._hist.iloc[rows, cols]
        out._meta = self._meta[cols]
        return out

    @property
    def hist(self):
        return self._hist.astype(float, copy=True)

    @hist.setter
    def hist(self, value: pd.DataFrame) -> None:
        self._validateHist(value)
        if len(value.columns) != len(self._meta):
            raise ValueError(
                "passed history does not match existing meta. "
                "To use a new `hist` with new `meta` use "
                "'History.createFromData(new_hist, new_meta)'"
            )
        self._hist = value.astype("category", copy=True)

    @property
    def meta(self) -> list[dict[str, Any]]:
        return list(self._meta)

    @meta.setter
    def meta(self, value: list[dict[str, Any]]) -> None:
        self._validateMetaList(value, self._hist)
        self._meta = _copy.deepcopy(value)

    @property
    def index(self) -> pd.Index:
        """
        The index of FH.

        The index is the same for all columns.

        Notes
        -----
        The index should always be equal to the flags series,
        the FH is associated with. If this is messed up
        something went wrong in saqc internals or in a user-
        defined test.

        Returns
        -------
        index : pd.Index
        """
        return self._hist.index

    @property
    def columns(self) -> pd.Index:
        """
        Columns of the FH.

        The columns are always continuously
        increasing integers, starting from 0.

        Returns
        -------
        columns : pd.Index
        """
        return self._hist.columns

    @property
    def empty(self) -> bool:
        """
        Indicator whether History is empty.

        True if History is entirely empty (no items).

        Returns
        -------
        bool
            If History is empty, return True, if not return False.
        """
        return len(self) == 0

    def _insert(self, s: pd.Series, pos: int) -> History:
        """
        Insert data at an arbitrary position in the FH.

        No validation of series is done here.

        Parameters
        ----------
        s : pd.Series
            the series to insert

        pos : int
            the position to insert

        Returns
        -------
        History
        """
        # Note:
        # all following code must handle a passed empty series

        # ensure continuous increasing columns
        assert 0 <= pos <= len(self.columns)
        self._hist[pos] = s.astype("category")
        return self

    def append(
        self, value: pd.Series | History, meta: dict[str, Any] | None = None
    ) -> History:
        """
        Create a new FH column and insert given pd.Series to it.

        Parameters
        ----------
        value : pd.Series or History
            The data to append. Must have dtype float and the index must
            match the index of the History.

        meta : dict, default None
            metadata dictionary to store with the series. Ignored if ``value`` is of
             type History. None defaults to a empty dictionary.

        Returns
        -------
        history with appended series

        Raises
        ------
        TypeError: if value is not pd.Series
        ValueError: on index miss-match or wrong dtype
        """
        if isinstance(value, History):
            return self._appendHistory(value)

        if not isinstance(value, pd.Series):
            raise TypeError("'value' is not a pd.Series")

        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError("'meta' must be of type None or dict")

        val = self._validateValue(value)
        if not val.index.equals(self.index):
            raise ValueError("Index does not match")

        self._insert(val, pos=len(self))
        self._meta.append(meta.copy())
        return self

    def _appendHistory(self, value: History) -> History:
        """
        Append multiple columns of a history to self.

        Parameters
        ----------
        value : History
            Holding the columns to append

        Returns
        -------
        History with appended columns.

        Raises
        ------
        ValueError : If the index of the passed history does not match.

        Notes
        -----
        This ignores the column names of the passed History.
        """
        self._validate(value._hist, value._meta)
        if not value.index.equals(self.index):
            raise ValueError("Index does not match")

        # we copy shallow because we only want to set new columns
        # the actual data copy happens in calls to astype
        value_hist = value._hist.copy(deep=False)
        value_meta = value._meta.copy()

        # rename columns, to avoid ``pd.DataFrame.loc`` become confused
        n = len(self.columns)
        columns = pd.Index(range(n, n + len(value_hist.columns)))
        value_hist.columns = columns

        hist = self._hist.astype(float)
        hist.loc[:, columns] = value_hist.astype(float)
        self._hist = hist.astype("category")
        self._meta += value_meta
        return self

    def squeeze(
        self, raw: bool = False, start: int | None = None, end: int | None = None
    ) -> pd.Series:
        """
        Reduce history to a series, by taking the last set value per row.

        By passing `start` and/or `end` only a slice of the history is used.
        This can be used to get the values of an earlier test. See the
        Examples.

        Parameters
        ----------
        raw : bool, default False
            If True, 'unset' values are represented by `nan`,
            otherwise, 'unset' values are represented by the
            `UNFLAGGED` (`-inf`) constant

        start : int, default None
            The first history column to use (inclusive).

        end : int, default None
            The last history column to use (exclusive).

        Returns
        -------
        pandas.Series

        Examples
        --------
        >>> from saqc.core.history import History
        >>> s0 = pd.Series([np.nan, np.nan, 99.])
        >>> s1 = pd.Series([1., 1., np.nan])
        >>> s2 = pd.Series([2., np.nan, 2.])
        >>> h = History(pd.Index([0,1,2])).append(s0).append(s1).append(s2)
        >>> h
             0    1    2
        0   nan  1.0  2.0
        1   nan  1.0  nan
        2  99.0  nan  2.0

        Get current flags.

        >>> h.squeeze()
        0    2.0
        1    1.0
        2    2.0
        dtype: float64

        Get only the flags that the last function had set:

        >>> h.squeeze(start=-1)
        0    2.0
        1   -inf
        2    2.0
        dtype: float64

        Get the flags before the last function run:

        >>> h.squeeze(end=-1)
        0     1.0
        1     1.0
        2    99.0
        dtype: float64

        Get only the flags that the 2nd function had set:

        >>> h.squeeze(start=1, end=2)
        0    1.0
        1    1.0
        2   -inf
        dtype: float64
        """
        hist = self._hist.iloc[:, slice(start, end)].astype(float)
        if hist.empty:
            result = pd.Series(data=np.nan, index=self._hist.index, dtype=float)
        else:
            result = AGGRGEGATIONS[AGGREGATION](hist)
        if not raw:
            result = result.fillna(UNFLAGGED)
        result.name = None
        return result

    def reindex(
        self, index: pd.Index, fill_value_last: float = UNFLAGGED, copy: bool = True
    ) -> History:
        """
        Reindex the History. Be careful this alters the past.

        Parameters
        ----------
        index : pd.Index
            the index to reindex to.

        fill_value_last : float, default UNFLAGGED
            value to fill nan's in the last column.
            Defaults to 0 (UNFLAGGED).

        copy : bool, default True
            If False, alter the underlying history, otherwise return a copy.

        Returns
        -------
        History
        """
        # Note: code must handle empty frames
        out = self.copy() if copy else self
        hist = out._hist.astype(float).reindex(index=index, copy=False)
        hist.iloc[:, -1:] = hist.iloc[:, -1:].fillna(fill_value_last)
        out._hist = hist.astype("category")
        return out

    def apply(
        self,
        index: pd.Index,
        func: Callable,
        func_kws: dict,
        func_handle_df: bool = False,
        copy: bool = True,
    ) -> History:
        """
        Apply a function on each column in history.

        The function must take a `pd.Series` as first arg, which is a column from
        `hist`. If ``func_handle_df=True`` each functions take a ``pd.DataFrame``
        as first argument, holding all columns at once.
        Bear in mind:
        - the functions mustn't alter the passed objects
        - the functions are not allowed to add or remove columns
        - the function must return same type as first argument
        - the returned object must have same index as the passed ``index`` to ``apply`` as first argument

        Parameters
        ----------
        index: pd.Index
            Index the new history should have. This is used to ensure the passed
            functions worked correct and also used if the function does not apply,
            because the initial history is empty. Then the altered empty history is
            reindexed to this index.

        func : callable
            function to apply on `History.hist` (flags DataFrame)

        func_kws : dict
            hist-function keywords dict

        func_handle_df : bool, default False
            If `True`, the Dataframe under `History`.hist is passed to the given functions,
            thus the function must handle `pd.Dataframes` as first input. If `False`, each
            column is passed separately, thus the function must handle those.

        copy : bool, default True
            If False, alter the underlying history, otherwise return a copy.


        Returns
        -------
        History with altered columns
        """
        hist = pd.DataFrame(index=index)

        # convert data to floats as functions may fail with categorical dtype
        if func_handle_df:
            hist = func(self._hist.astype(float, copy=True), **func_kws)
        else:
            for pos in self.columns:
                hist[pos] = func(self._hist[pos].astype(float, copy=True), **func_kws)

        try:
            self._validate(hist, self._meta)
        except Exception as e:
            raise ValueError(
                f"result from applied function is not a valid History, because {e}"
            ) from e

        if copy:
            history = History(index=None)  # noqa
            history._meta = self._meta.copy()
        else:
            history = self

        history._hist = hist.astype("category")

        return history

    def copy(self, deep=True) -> History:
        """
        Make a copy of the FH.

        Parameters
        ----------
        deep : bool, default True
            - ``True``: make a deep copy
            - ``False``: make a shallow copy

        Returns
        -------
        copy : History
            the copied FH
        """
        copyfunc = _copy.deepcopy if deep else _copy.copy
        new = History(self.index)
        new._hist = self._hist.copy(deep)
        new._meta = copyfunc(self._meta)
        return new

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    def __len__(self) -> int:
        return len(self._hist.columns)

    def __repr__(self):
        if self.empty:
            return str(self._hist).replace("DataFrame", "History")

        r = self._hist.astype(str)

        return str(r)[1:]

    # --------------------------------------------------------------------------------
    # validation
    #

    @classmethod
    def _validate(
        cls, hist: pd.DataFrame, meta: List[Any]
    ) -> Tuple[pd.DataFrame, List]:
        """
        check type, columns, index, dtype of hist and if the meta fits also
        """
        cls._validateHist(hist)
        cls._validateMetaList(meta, hist)
        return hist, meta

    @classmethod
    def _validateHist(cls, obj):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"'hist' must be of type pd.DataFrame, "
                f"but is of type {type(obj).__name__}"
            )
        if not obj.columns.equals(pd.RangeIndex(len(obj.columns))):
            raise ValueError(
                "Columns of 'hist' must consist of "
                "continuous increasing integers, "
                "starting with 0."
            )
        for c in obj.columns:
            try:
                cls._validateValue(obj[c])
            except Exception as e:
                raise ValueError(f"Bad column in hist. column '{c}': {e}") from None
        return obj

    @classmethod
    def _validateMetaList(cls, obj, hist=None):
        if not isinstance(obj, list):
            raise TypeError(
                f"'meta' must be of type list, got type {type(obj).__name__}"
            )
        if hist is not None:
            if not len(obj) == len(hist.columns):
                raise ValueError(
                    "'meta' must have as many entries as columns in 'hist'"
                )
        for i, item in enumerate(obj):
            try:
                cls._validateMetaDict(item)
            except Exception as e:
                raise ValueError(f"Bad meta. item {i}: {e}") from None
        return obj

    @staticmethod
    def _validateMetaDict(obj):
        if not isinstance(obj, dict):
            raise TypeError("obj must be dict")
        if not all(isinstance(k, str) for k in obj.keys()):
            raise ValueError("all keys in dict must be strings")
        return obj

    @staticmethod
    def _validateValue(obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(
                f"value must be of type pd.Series, got type {type(obj).__name__}"
            )
        if not is_float_dtype(obj.dtype) and not isinstance(
            obj.dtype, pd.CategoricalDtype
        ):
            raise ValueError("dtype must be float or categorical")
        return obj

    @classmethod
    def createFromData(cls, hist: pd.DataFrame, meta: List[Dict], copy: bool = False):
        """
        Create a History from existing data.

        Parameters
        ----------
        hist : pd.Dataframe
            Data that define the flags of the history.

        meta : List of dict
            A list holding meta information for each column, therefore it must
            have the same number of entries as columns exist in `hist`.

        copy : bool, default False
            If `True`, the input data is copied, otherwise not.


        Notes
        -----
        To create a very simple History from a flags dataframe ``f`` use
        ``mask = pd.DataFrame(True, index=f.index, columns=f.columns``
        and
        ``meta = [{}] * len(f.columns)``.

        Returns
        -------
        History
        """
        cls._validate(hist, meta)

        if copy:
            hist = hist.copy()
            meta = _copy.deepcopy(meta)

        history = cls(index=None)  # noqa
        history._hist = hist.astype("category", copy=False)
        history._meta = meta
        return history


def createHistoryFromData(
    hist: pd.DataFrame,
    meta: List[Dict],
    copy: bool = False,
):
    """
    Create a History from existing data.

    Parameters
    ----------
    hist : pd.Dataframe
        Data that define the flags of the history.

    meta : List of dict
        A list holding meta information for each column, therefore it must
        have the same number of entries as columns exist in `hist`.

    copy : bool, default False
        If `True`, the input data is copied, otherwise not.


    Notes
    -----
    To create a very simple History from a flags dataframe ``f`` use
    ``mask = pd.DataFrame(True, index=f.index, columns=f.columns``
    and
    ``meta = [{}] * len(f.columns)``.

    Returns
    -------
    History
    """
    # todo: expose History, enable this warning
    # warnings.warn(
    #     "saqc.createHistoryFromData() will be deprecated soon. "
    #     "Please use saqc.History.createFromData() instead.",
    #     category=FutureWarning,
    # )
    return History.createFromData(hist, meta, copy)
