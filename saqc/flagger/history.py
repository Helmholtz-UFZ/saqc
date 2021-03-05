#!/usr/bin/env python
from __future__ import annotations

from typing import Tuple, Type
import pandas as pd
import numpy as np
from saqc.common import *


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
            hist = self._validate_hist(hist)
            mask = pd.DataFrame(True, index=hist.index, columns=hist.columns)

        else:
            hist, mask = self._validate_hist_with_mask(hist, mask)

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

    def append(self, value: pd.Series, force=False) -> History:
        """
        Create a new FH column and insert given pd.Series to it.

        Parameters
        ----------
        value : pd.Series
            the data to append. Must have dtype float and the index must
            match the index of the FH.

        force : bool, default False
            if True the internal mask is updated in a way that the currently
            set value (series values) will be returned if ``History.max()``
            is called. This apply for all valid values (not ``np.Nan`` and
            not ``-np.inf``).

        Raises
        ------
        ValueError: on index miss-match or wrong dtype
        TypeError: if value is not pd.Series

        Returns
        -------
        History: FH with appended series
        """
        s = self._validate_value(value)

        if len(self) > 0 and not s.index.equals(self.index):
            raise ValueError("Index must be equal to FlagHistory index")

        self._insert(value, pos=len(self), force=force)
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

    def _validate_hist_with_mask(self, obj: pd.DataFrame, mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        check type, columns, index, dtype and if the mask fits the obj.
        """

        # check hist
        self._validate_hist(obj)

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

    def _validate_hist(self, obj: pd.DataFrame) -> pd.DataFrame:
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

    def _validate_value(self, obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(f'value must be of type pd.Series, but {type(obj).__name__} was given')

        if not obj.dtype == float:
            raise ValueError('dtype must be float')

        return obj
