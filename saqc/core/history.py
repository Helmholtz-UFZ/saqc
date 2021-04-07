#!/usr/bin/env python
from __future__ import annotations

from typing import Tuple, Type, Union
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
    the FH is empty and has no columns at all. If an initial `UNFLAGGED`-
    column is desired, it must be created manually, or passed via the ``hist``
    parameter. The same way a new FH can be created from an existing one.

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
    hist : pd.Dataframe, default None
        if None a empty FH is created, otherwise the existing dataframe
        is taken as the initial history.

    mask : pd.Dataframe, default None
        a mask holding the boolean force values. It must match the passed
        ``hist``. If None an matching mask is created, assuming force never
        was passed to any test.

    copy : bool, default False
        If True, the input data is copied, otherwise not.
    """

    def __init__(self, hist: pd.DataFrame = None, mask: pd.DataFrame = None, copy: bool = False):

        # this is a hidden _feature_ and not exposed by the type
        # of the hist parameter and serve as a fastpath for internal
        # fast creation of a new FH, where no checks are needed.
        if isinstance(hist, History):
            # keep this order, otherwise hist.mask
            # will refer to pd.Dataframe.mask
            mask = hist.mask
            hist = hist.hist

        elif hist is None and mask is None:
            hist = pd.DataFrame()
            mask = pd.DataFrame()

        elif hist is None and mask is not None:
            raise ValueError("Cannot take 'mask' without 'hist'")

        elif hist is not None and mask is None:
            hist = self._validateHist(hist)
            mask = pd.DataFrame(True, index=hist.index, columns=hist.columns)

        else:
            hist, mask = self._validateHistWithMask(hist, mask)

        if copy:
            hist = hist.copy()
            mask = mask.copy()

        self.hist = hist
        self.mask = mask

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

        self.hist[pos] = s

        return self

    def append(self, value: Union[pd.Series, History], force=False) -> History:
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

        value = self._validateValue(value)
        if len(self) > 0 and not value.index.equals(self.index):
            raise ValueError("Index does not match")

        self._insert(value, pos=len(self), force=force)
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
        self._validateHistWithMask(value.hist, value.mask)
        if len(self) > 0 and not value.index.equals(self.index):
            raise ValueError("Index does not match")

        n = len(self.columns)
        value_hist = value.hist
        value_mask = value.mask

        if not force:
            value_hist = value_hist.iloc[:, n:]
            value_mask = value_mask.iloc[:, n:]

        # rename columns, to avoid ``pd.DataFrame.loc`` become confused
        columns = pd.Index(range(n, len(value_hist.columns) + 1))
        value_hist.columns = columns
        value_mask.columns = columns

        self.hist.loc[:, columns] = value_hist.copy()
        self.mask.loc[:, columns] = value_mask.copy()
        return self

    def squeeze(self, n: int) -> History:
        """
        Squeeze last `n` columns to a single column.

        This **not** changes the result of ``History.max()``.

        Parameters
        ----------
        n : int
            last n columns to squeeze

        Notes
        -----
        The new column number (column name) will be the lowest of
        the squeezed. This ensure that the column numbers are always
        monotonically increasing.

        Bear in mind, this works inplace, if a copy is needed, call ``copy`` before.

        Returns
        -------
        History
            squeezed history
        """
        if n <= 1:
            return self

        if n > len(self):
            raise ValueError(f"'n={n}' cannot be greater than columns in the FH")

        # calc the squeezed series.
        # we dont have to care about any forced series
        # because anytime force was given, the False's in
        # the mask were propagated back over the whole FH
        mask = self.mask.iloc[:, -n:]
        hist = self.hist.iloc[:, -n:]
        s = hist[mask].max(axis=1)

        # slice self down
        # this may leave us in an unstable state, because
        # the last column maybe is not entirely True, but
        # the following append, will fix this
        self.hist = self.hist.iloc[:, :-n]
        self.mask = self.mask.iloc[:, :-n]

        self.append(s)
        return self

    def idxmax(self) -> pd.Series:
        """
        Get the index of the maximum value per row of the FH.

        Returns
        -------
        pd.Series: maximum values
        """
        return self.hist[self.mask].idxmax(axis=1)

    def max(self) -> pd.Series:
        """
        Get the maximum value per row of the FH.

        Returns
        -------
        pd.Series: maximum values
        """
        return self.hist[self.mask].max(axis=1)

    @property
    def _constructor(self) -> Type['History']:
        return History

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
        return self._constructor(hist=self, copy=deep)

    def reindex(self, index: pd.Index, fill_value_last: float = UNFLAGGED) -> History:
        """
        Reindex the History. Be careful this alters the past.

        Parameters
        ----------
        index : pd.Index
            the index to reindex to.
        fill_value_last : float, default UNFLAGGED
            value to fill nan's (UNTOUCHED) in the last column. Defaults to 0 (UNFLAGGED).

        Returns
        -------
        History
        """
        self.hist = self.hist.reindex(index=index, copy=False, fill_value=np.nan)
        self.mask = self.mask.reindex(index=index, copy=False, fill_value=False)
        # Note: all following code must handle empty frames
        self.hist.iloc[:, -1:] = self.hist.iloc[:, -1:].fillna(fill_value_last)
        self.mask.iloc[:, -1:] = True
        return self

    def __copy__(self, deep: bool = True):
        return self.copy(deep=deep)

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
            return str(self.hist).replace('DataFrame', 'History')

        repr = self.hist.astype(str)
        m = self.mask

        repr[m] = ' ' + repr[m] + ' '
        repr[~m] = '(' + repr[~m] + ')'

        return str(repr)[1:]

    # --------------------------------------------------------------------------------
    # validation
    #

    @staticmethod
    def _validateHistWithMask(obj: pd.DataFrame, mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        check type, columns, index, dtype and if the mask fits the obj.
        """

        # check hist
        History._validateHist(obj)

        # check mask
        if not isinstance(mask, pd.DataFrame):
            raise TypeError(f"'mask' must be of type pd.DataFrame, but {type(mask).__name__} was given")

        if any(mask.dtypes != bool):
            raise ValueError("dtype of all columns in 'mask' must be bool")

        if not mask.empty and not mask.iloc[:, -1].all():
            raise ValueError("the values in the last column in mask must be 'True' everywhere.")

        # check combination of hist and mask
        if not obj.columns.equals(mask.columns):
            raise ValueError("'hist' and 'mask' must have same columns")

        if not obj.index.equals(mask.index):
            raise ValueError("'hist' and 'mask' must have same index")

        return obj, mask

    @staticmethod
    def _validateHist(obj: pd.DataFrame) -> pd.DataFrame:
        """
        check type, columns, dtype of obj.
        """

        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"'hist' must be of type pd.DataFrame, but {type(obj).__name__} was given")

        if any(obj.dtypes != float):
            raise ValueError('dtype of all columns in hist must be float')

        if not obj.empty and (
                not obj.columns.equals(pd.Index(range(len(obj.columns))))
                or obj.columns.dtype != int
        ):
            raise ValueError("column names must be continuous increasing int's, starting with 0.")

        return obj

    @staticmethod
    def _validateValue(obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(f'value must be of type pd.Series, but {type(obj).__name__} was given')

        if not obj.dtype == float:
            raise ValueError('dtype must be float')

        return obj


def applyFunctionOnHistory(
        history: History,
        hist_func: callable,
        hist_kws: dict,
        mask_func: callable,
        mask_kws: dict,
        last_column: Union[pd.Series, Literal['dummy'], None] = None,
        func_handle_df: bool = False,
):
    """
    Apply function on each column in history.

    Two functions must be given. Both are called for each column in the History unless ``func_handle_df=True`` is
    given. One is called on ``History.hist``, the other on ``History.mask``.
    Both function must take a pd.Series as first arg, which is the column from hist or mask respectively. If
    ``func_handle_df=True`` each functions must take a ``pd.DataFrame`` as first argument, holding all columns
    at once. The function must return same type as first argument.

    Parameters
    ----------
    history : History
        History object to alter
    hist_func : callable
        function to apply on `History.hist` (flags DataFrame)
    hist_kws : dict
        hist-function keywords dict
    mask_func : callable
        function to apply on `History.mask` (force mask DataFrame)
    mask_kws : dict
        mask-function keywords dict
    last_column : pd.Series or None, default None
        The last column to apply. If None, no extra column is appended.
    func_handle_df : bool
        If `True`, the whole History{.hist, .mask} are passed to the given functions, thus the
        function must handle `pd.Dataframes` as first input. If `False`, each column is passed
        separately, thus the functions must handle those.

    Notes
    -----
    After the functions are called, all `NaN`'s in `History.mask` are replaced with `False`,
    and the `.mask` is casted to bool, to ensure a consistent History.

    Returns
    -------
    history with altered columns
    """
    new_history = History()

    if func_handle_df:
        history.hist = hist_func(history.hist, **hist_kws)
        history.mask = hist_func(history.mask, **mask_kws)

    else:
        for pos in history.columns:
            new_history.hist[pos] = hist_func(history.hist[pos], **hist_kws)
            new_history.mask[pos] = mask_func(history.mask[pos], **mask_kws)

    # handle unstable state
    if last_column is None:
        new_history.mask.iloc[:, -1:] = True
    else:
        if isinstance(last_column, str) and last_column == 'dummy':
            last_column = pd.Series(UNTOUCHED, index=new_history.index, dtype=float)

        new_history.append(last_column, force=True)

    # assure a boolean mask
    new_history.mask = new_history.mask.fillna(False).astype(bool)
    new_history.hist[0] = UNFLAGGED

    return new_history

