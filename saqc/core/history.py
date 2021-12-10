#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy, copy as shallowcopy
from typing import Dict, Tuple, Union, List, Any

import pandas as pd
import numpy as np

from saqc.constants import UNFLAGGED


class History:
    """
    Saqc internal storage for the history of a (single) flags column.

    The flag-history (FH) stores the history of a flags column. Each time
    ``append`` is called a new column is appended to the FH. The column
    names are increasing integers starting with 0. After initialisation
    the FH is empty and has no columns at all.

    To get the worst flags (highest value) that are currently stored in
    the FH, we provide a ``max()`` method. It returns a pd.Series indicating
    the worst flag per row.

    For more details and a detailed discussion, why this is needed, how this
    works and possible other implementations, see #GL143 [1].

    [1] https://git.ufz.de/rdm-software/saqc/-/issues/143

    Parameters
    ----------
    index: pd.Index
        A index that fit the flags to be insert.

    See Also
    --------
    createHistoryFromData: function to create History from existing data
    """

    def __init__(self, index: pd.Index):

        self.hist = pd.DataFrame(index=index)
        self.meta = []

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
        return self.hist.index

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
        return self.hist.columns

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
        assert 0 <= pos <= len(self)

        self.hist[pos] = s.astype("category")

        return self

    def append(self, value: Union[pd.Series, History], meta: dict = None) -> History:
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

        if not isinstance(meta, dict):
            raise TypeError("'meta' must be of type None or dict")

        val = self._validateValue(value)
        if not val.index.equals(self.index):
            raise ValueError("Index does not match")

        self._insert(val, pos=len(self))
        self.meta.append(meta.copy())
        return self

    def _appendHistory(self, value: History):
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
        self._validate(value.hist, value.meta)
        if not value.index.equals(self.index):
            raise ValueError("Index does not match")

        # we copy shallow because we only want to set new columns
        # the actual data copy happens in calls to astype
        value_hist = value.hist.copy(deep=False)
        value_meta = value.meta.copy()

        # rename columns, to avoid ``pd.DataFrame.loc`` become confused
        n = len(self.columns)
        columns = pd.Index(range(n, n + len(value_hist.columns)))
        value_hist.columns = columns

        hist = self.hist.astype(float)
        hist.loc[:, columns] = value_hist.astype(float)
        self.hist = hist.astype("category")
        self.meta += value_meta
        return self

    def max(self, raw=False) -> pd.Series:
        """
        Get the maximum value per row of the FH.

        Returns
        -------
        pd.Series: maximum values
        """
        result = self.hist.astype(float)
        if result.empty:
            result = pd.DataFrame(data=np.nan, index=self.hist.index, columns=[0])

        result = result.ffill(axis=1).iloc[:, -1]

        if raw:
            return result
        else:
            return result.fillna(UNFLAGGED)

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
        out = self.copy() if copy else self

        hist = out.hist.astype(float).reindex(
            index=index, copy=False, fill_value=np.nan
        )

        # Note: all following code must handle empty frames
        hist.iloc[:, -1:] = hist.iloc[:, -1:].fillna(fill_value_last)

        out.hist = hist.astype("category")

        return out

    def apply(
        self,
        index: pd.Index,
        func: callable,
        func_kws: dict,
        func_handle_df: bool = False,
        copy: bool = True,
    ):
        """
        Apply a function on each column in history.

        The function must take a `pd.Series` as first arg, which is a column from
        `hist`. If ``func_handle_df=True`` each functions take a ``pd.DataFrame``
        as first argument, holding all columns at once.
        Bear in mind:
        - the functions mustn't alter the passed objects
        - the functions are not allowed to add or remove columns
        - the function must return same type as first argument
        - the returned object must have same index as the passed ``index`` to ``apply``
            as first argument

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
        history with altered columns
        """
        hist = pd.DataFrame(index=index)

        # implicit copy by astype
        # convert data to floats as functions may fail with categoricals
        if func_handle_df:
            hist = func(self.hist.astype(float), **func_kws)
        else:
            for pos in self.columns:
                hist[pos] = func(self.hist[pos].astype(float), **func_kws)

        History._validate(hist, self.meta)

        if copy:
            history = History(index=None)  # noqa
            history.meta = self.meta.copy()
        else:
            history = self

        history.hist = hist.astype("category")

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
        copyfunc = deepcopy if deep else shallowcopy
        new = History(self.index)
        new.hist = self.hist.copy(deep)
        new.meta = copyfunc(self.meta)
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
        return len(self.hist.columns)

    def __repr__(self):

        if self.empty:
            return str(self.hist).replace("DataFrame", "History")

        r = self.hist.astype(str)

        return str(r)[1:]

    # --------------------------------------------------------------------------------
    # validation
    #

    @staticmethod
    def _validate(hist: pd.DataFrame, meta: List[Any]) -> Tuple[pd.DataFrame, List]:
        """
        check type, columns, index, dtype of hist and if the meta fits also
        """

        # check hist
        if not isinstance(hist, pd.DataFrame):
            raise TypeError(
                f"'hist' must be of type pd.DataFrame, but is of type {type(hist).__name__}"
            )
        # isin([float, ..]) does not work !
        if not (
            (hist.dtypes == float)
            | (hist.dtypes == np.float32)
            | (hist.dtypes == np.float64)
            | (hist.dtypes == "category")
        ).all():
            raise ValueError(
                "dtype of all columns in hist must be float or categorical"
            )

        if not hist.empty and (
            not hist.columns.equals(pd.Index(range(len(hist.columns))))
            or not np.issubdtype(hist.columns.dtype, np.integer)
        ):
            raise ValueError(
                "column names must be continuous increasing int's, starting with 0."
            )

        # check meta
        if not isinstance(meta, list):
            raise TypeError(
                f"'meta' must be of type list, but is of type {type(meta).__name__}"
            )
        if not all([isinstance(e, dict) for e in meta]):
            raise TypeError("All elements in meta must be of type 'dict'")

        # check combinations of hist and meta
        if not len(hist.columns) == len(meta):
            raise ValueError(
                "'meta' must have as many entries as columns exist in hist"
            )

        return hist, meta

    @staticmethod
    def _validateValue(obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(
                f"value must be of type pd.Series, but {type(obj).__name__} was given"
            )

        if not ((obj.dtype == float) or isinstance(obj.dtype, pd.CategoricalDtype)):
            raise ValueError("dtype must be float or categorical")

        return obj


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
    History._validate(hist, meta)

    if copy:
        hist = hist.copy()
        meta = deepcopy(meta)

    history = History(index=None)  # noqa
    history.hist = hist.astype("category", copy=False)
    history.meta = meta
    return history
