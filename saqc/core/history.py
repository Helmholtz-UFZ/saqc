#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy, copy
import itertools

from typing import Dict, Tuple, Type, Union, List, Any
from typing_extensions import Literal
import pandas as pd
import numpy as np

from saqc.constants import *


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

    To counteract the problem, that one may want to force a better flag
    value than the current worst, ``append`` provide a ``force`` keyword.
    Internal we need to store the force information in an additional mask.

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
        self.mask = pd.DataFrame(index=index)
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
        # we take self.mask here, because it cannot have NaN's,
        # but self.hist could have -> see pd.DataFrame.empty
        return self.mask.empty

    def _insert(self, s: pd.Series, pos: int, force=False) -> History:
        """
        Insert data at an arbitrary position in the FH.

        No validation of series is done here.

        Parameters
        ----------
        s : pd.Series
            the series to insert

        pos : int
            the position to insert

        force : bool, default False
            if True the internal mask is updated accordingly that the values overwrite
            any earlier values in the FH.

        Returns
        -------
        History
        """
        # Note:
        # all following code must handle a passed empty series

        # ensure continuous increasing columns
        assert 0 <= pos <= len(self)

        if pos == len(self):  # append
            self.mask[pos] = pd.Series(True, index=s.index, dtype=bool)

        if force:
            touched = s.notna()
            self.mask.iloc[touched, :pos] = False

        self.hist[pos] = s.astype("category")

        return self

    def append(
        self, value: Union[pd.Series, History], force: bool = False, meta: dict = None
    ) -> History:
        """
        Create a new FH column and insert given pd.Series to it.

        Parameters
        ----------
        value : pd.Series or History
            The data to append. Must have dtype float and the index must
            match the index of the History.

        force : bool, default False

            If ``value`` is a ``pd.Series``:

                - ``force=True`` : the internal mask is updated in a way that the currently
                  set values gets the highest priority in the current history.
                  This means, these values are guaranteed to be returned if ``History.max()``
                  is called, until newer possibly higher flags are set. Bear in mind
                  that this never apply for `Nan`-values.
                - ``force=False`` : values are not treated special.

            If ``value`` is a ``History``:

                - ``force=True`` : All columns are appended to the existing history.
                - ``force=False`` : Only columns that are `newer` are appended. This means
                  the first ``N`` columns of the passed history are discarded, where ``N`` is the
                  number of existing columns in the current history.

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
            return self._appendHistory(value, force=force)

        if not isinstance(value, pd.Series):
            raise TypeError("'value' is not a pd.Series")

        if meta is None:
            meta = {}

        if not isinstance(meta, dict):
            raise TypeError("'meta' must be of type None or dict")

        val = self._validateValue(value)
        if not val.index.equals(self.index):
            raise ValueError("Index does not match")

        self._insert(val, pos=len(self), force=force)
        self.meta.append(deepcopy(meta))
        return self

    def _appendHistory(self, value: History, force: bool = False):
        """
        Append multiple columns of a history to self.

        Parameters
        ----------
        value : History
            Holding the columns to append
        force : bool
            If False, the first `N` columns in the passed History are discarded, where
            `N` is the number of columns in the original history.
            If ``force=True`` all columns are appended.

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
        self._validate(value.hist, value.mask, value.meta)
        if not value.index.equals(self.index):
            raise ValueError("Index does not match")

        n = len(self.columns)
        # don't overwrite the `.columns` of the input down the line
        value_hist = value.hist.copy(deep=False)
        value_mask = value.mask.copy(deep=False)
        value_meta = deepcopy(value.meta)

        if not force:
            value_hist = value_hist.iloc[:, n:]
            value_mask = value_mask.iloc[:, n:]
            value_meta = value_meta[n:]

        # rename columns, to avoid ``pd.DataFrame.loc`` become confused
        columns = pd.Index(range(n, n + len(value_hist.columns)))
        value_hist.columns = columns
        value_mask.columns = columns

        # clear the current mask
        self.mask.loc[(~value_mask & value_hist.notna()).any(axis="columns")] = False

        hist = self.hist.astype(float)
        hist.loc[:, columns] = value_hist.astype(float)
        self.hist = hist.astype("category", copy=True)
        self.mask.loc[:, columns] = value_mask.copy()
        self.meta += value_meta
        return self

    def idxmax(self) -> pd.Series:
        """
        Get the index of the maximum value per row of the FH.

        Returns
        -------
        pd.Series: maximum values
        """
        if self.mask.empty:
            return pd.Series(np.nan, index=self.index)
        return self.hist[self.mask].astype(float).idxmax(axis=1)

    def max(self, raw=False) -> pd.Series:
        """
        Get the maximum value per row of the FH.

        Returns
        -------
        pd.Series: maximum values
        """
        result = self.hist[self.mask].max(axis=1)
        if raw:
            return result
        else:
            return result.fillna(UNFLAGGED)

    def reindex(self, index: pd.Index, fill_value_last: float = UNFLAGGED) -> History:
        """
        Reindex the History. Be careful this alters the past.

        Parameters
        ----------
        index : pd.Index
            the index to reindex to.
        fill_value_last : float, default UNFLAGGED
            value to fill nan's (UNTOUCHED) in the last column.
            Defaults to 0 (UNFLAGGED).

        Returns
        -------
        History
        """
        hist = self.hist.astype(float).reindex(
            index=index, copy=False, fill_value=np.nan
        )
        mask = self.mask.astype(bool).reindex(index=index, copy=False, fill_value=False)

        # Note: all following code must handle empty frames
        hist.iloc[:, -1:] = hist.iloc[:, -1:].fillna(fill_value_last)
        mask.iloc[:, -1:] = True

        self.mask = mask.astype(bool)
        self.hist = hist.astype("category")

        return self

    def apply(
        self,
        index: pd.Index,
        hist_func: callable,
        hist_kws: dict,
        mask_func: callable,
        mask_kws: dict,
        func_handle_df: bool = False,
        copy: bool = True,
    ):
        """
        Apply a function on each column in history.

        Two functions must be given. Both are called for each column in the History
        unless ``func_handle_df=True`` is given. One is called on ``History.hist``,
        the other on ``History.mask``. Both function must take a `pd.Series` as first
        arg, which is a column from `hist` respectively `mask`. If
        ``func_handle_df=True`` each functions take a ``pd.DataFrame`` as first
        argument, holding all columns at once.
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

        hist_func : callable
            function to apply on `History.hist` (flags DataFrame)

        hist_kws : dict
            hist-function keywords dict

        mask_func : callable
            function to apply on `History.mask` (force mask DataFrame)

        mask_kws : dict
            mask-function keywords dict

        func_handle_df : bool, default False
            If `True`, the Dataframe under `History`.hist, respectively  `History.mask`
            is passed to the given functions, thus both(!) function must handle
            `pd.Dataframes` as first input. If `False`, each column is passed separately,
            thus the functions must handle those.

        copy : bool, default True
            If False, alter the underlying history, otherwise return a copy.

        Notes
        -----
        After the functions are called, all `NaN`'s in `History.mask` are replaced by
        `False`, and the `.mask` is casted to bool, to ensure a consistent History.

        Returns
        -------
        history with altered columns
        """
        hist = pd.DataFrame(index=index)
        mask = pd.DataFrame(index=index)

        if func_handle_df:
            # we need to pass the data as floats as functions may fail with Categorical
            hist = hist_func(self.hist.astype(float), **hist_kws)
            mask = mask_func(self.mask, **mask_kws)

        else:
            for pos in self.columns:
                hist[pos] = hist_func(self.hist[pos].astype(float), **hist_kws)
                mask[pos] = mask_func(self.mask[pos], **mask_kws)

        # handle unstable state
        mask.iloc[:, -1:] = True

        History._validate(hist, mask, self.meta)

        if copy:
            history = History(index=None)  # noqa
            history.meta = deepcopy(self.meta)
        else:
            history = self

        history.hist = hist.astype("category")
        history.mask = mask.fillna(True).astype(bool)

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
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def __len__(self) -> int:
        return len(self.hist.columns)

    def __repr__(self):

        if self.empty:
            return str(self.hist).replace("DataFrame", "History")

        r = self.hist.astype(str)
        m = self.mask

        r[m] = " " + r[m] + " "
        r[~m] = "(" + r[~m] + ")"

        return str(r)[1:]

    # --------------------------------------------------------------------------------
    # validation
    #

    @staticmethod
    def _validate(
        hist: pd.DataFrame, mask: pd.DataFrame, meta: List[Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        check type, columns, index, dtype of hist and mask and if the meta fits also.
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
            or hist.columns.dtype != int
        ):
            raise ValueError(
                "column names must be continuous increasing int's, starting with 0."
            )

        # check mask
        if not isinstance(mask, pd.DataFrame):
            raise TypeError(
                f"'mask' must be of type pd.DataFrame, but is of type {type(mask).__name__}"
            )

        if not (mask.dtypes == bool).all():
            raise ValueError("dtype of every columns in 'mask' must be bool")

        if not mask.empty and not mask.iloc[:, -1].all():
            raise ValueError(
                "the values in the last column in mask must be 'True' everywhere."
            )

        # check meta
        if not isinstance(meta, list):
            raise TypeError(
                f"'meta' must be of type list, but is of type {type(meta).__name__}"
            )
        if not all([isinstance(e, dict) for e in meta]):
            raise TypeError("All elements in meta must be of type 'dict'")

        # check combinations of hist and mask and meta
        if not hist.columns.equals(mask.columns):
            raise ValueError("'hist' and 'mask' must have the same columns")

        if not hist.index.equals(mask.index):
            raise ValueError("'hist' and 'mask' must have the same index")

        if not len(hist.columns) == len(meta):
            raise ValueError(
                "'meta' must have as many entries as columns exist in hist"
            )

        return hist, mask, meta

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
    mask: pd.DataFrame,
    meta: List[Dict],
    copy: bool = False,
):
    """
    Create a History from existing data.

    Parameters
    ----------
    hist : pd.Dataframe
        Data that define the flags of the history.

    mask : pd.Dataframe
        The mask holding the boolean force values. The following
        points must hold:

        * columns must be equal to the columns of `hist`
        * the last column must be entirely `True`
        * at most one change from False to True is allowed per row

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
    History._validate(hist, mask, meta)

    if copy:
        hist = hist.copy()
        mask = mask.copy()
        meta = deepcopy(meta)

    history = History(index=None)  # noqa
    history.hist = hist.astype("category", copy=False)
    history.mask = mask
    history.meta = meta
    return history
