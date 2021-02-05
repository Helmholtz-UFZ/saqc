#!/usr/bin/env python
from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


class Backtrack:

    def __init__(self, bt: pd.DataFrame = None, mask: pd.DataFrame = None, copy: bool = False):

        if bt is None and mask is not None:
            raise ValueError("Cannot take 'mask' with no 'bt'")

        if bt is None:
            bt = pd.DataFrame()
            mask = pd.DataFrame()

        if isinstance(bt, Backtrack):
            bt = bt.bt
            mask = bt.mask
        else:
            bt, mask = self._validate_bt(bt, mask)

        if copy:
            bt = bt.copy()
            mask = mask.copy()

        self.bt = bt
        self.mask = mask
        self._nr = len(bt)

    @property
    def nr(self):
        return self._nr

    @property
    def index(self) -> pd.Index:
        return self.bt.index

    def _validate_bt(self, obj: pd.DataFrame, mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for name, obj in zip(['BT', 'mask'], [obj, mask]):
            if not isinstance(obj, pd.DataFrame):
                raise TypeError(f'{name} must be of type pd.DataFrame, but {type(obj).__name__} was given')

        if any(obj.dtypes != float):
            raise ValueError('dtype of all columns in BT must be float')

        if any(mask.dtypes != bool):
            raise ValueError('dtype of all columns in mask must be bool')

        if not obj.columns.equals(mask.columns):
            raise ValueError("'BT' and 'mask' must have same columns")

        if not obj.index.equals(mask.columns):
            raise ValueError("'BT' and 'mask' must have same index")

        return obj, mask

    def _validate_value(self, obj: pd.Series) -> pd.Series:
        """
        index is not checked !
        """
        if not isinstance(obj, pd.Series):
            raise TypeError(f'value must be of type pd.Series, but {type(obj).__name__} was given')

        if not obj.dtype == float:
            raise ValueError('dtype must be float')

        return obj

    def _update_mask(self, touched: pd.Series):
        """
        updates internal mask by reference mask.

        Every row in the internal mask is set to False, where
        ``touched`` is True.

        Parameters
        ----------
        touched : pd.Series
            reference mask
        """
        self.bt[touched] = False

    def _insert(self, value: pd.Series, nr: int, force=False) -> Backtrack:
        """
        Insert data at an arbitrary position in the BT.

        No checks.

        Parameters
        ----------
        value : pd.Series
            the series to insert

        nr : int
            the position to insert

        force : bool, default False
            if True the internal mask is updated accordingly

        Returns
        -------
        Backtrack
        """
        s = self._validate_value(value)

        if s.empty:
            raise ValueError('Cannot insert empty pd.Series')

        if not self.bt.empty and s.index.equals(self.index):
            raise ValueError("Index must be equal to BT's index")

        if force:
            touched = np.isfinite(s)
            self._update_mask(touched)

        self.mask[nr] = pd.Series(True, index=s.index)
        self.bt[nr] = s

        return self

    def squeeze(self, n: int):
        """
        Squeeze last `n` columns to a single column.

        Parameters
        ----------
        n : int
            last n cloumns to squeeze

        Notes
        -----
        The new column number (column name) will be the lowest of
        the squeezed. This ensure that the column numbers are always
        monotonic increasing.

        Bear in mind, this works inplace, if a copy is needed, call ``copy`` before.

        Returns
        -------
        Backtrack: squeezed backtrack
        """
        if n == 0:
            return self

        if n > len(self):
            raise ValueError(f"'n={n}' cannot be greater than columns in the BT")

        # shortcut
        if len(self) == n:
            self.bt = pd.DataFrame()
            self.mask = pd.DataFrame()
            s = self.max()

        else:
            # calc the squeezed series.
            # we dont have to care about any forced series
            # because anytime force is given, the False's in
            # the mask are propagated back over the whole BT
            mask = self.mask.iloc[:, -n:]
            bt = self.bt.iloc[: -n:]
            s = bt[mask].max(axis=1)

            # slice self down
            self.bt = self.bt.iloc[:, :-n]
            self.mask = self.mask.iloc[:, :-n]

        self._nr = len(self)
        self.append(s)
        return self

    def append(self, value: pd.Series) -> Backtrack:
        """
        Create a new BT column and insert given pd.Series to it.

        Parameters
        ----------
        value : pd.Series
            the data to append. Must have dtype float and the index must
            match the index of the BT.

        Raises
        ------
        ValueError: on index miss-match or wrong dtype
        TypeError: if value is not pd.Series

        Returns
        -------
        Backtrack: BT with appended series
        """
        self._insert(value, self.nr + 1)
        self._nr += 1
        return self

    def max(self):
        """
        Get the maximum value per row of non-masked data.

        Returns
        -------
        pd.Series: maximum values
        """
        return self.bt[self.mask].max(axis=1)

    def __len__(self):
        return len(self.bt)

    def copy(self, deep=True):
        return Backtrack(bt=self, copy=deep)
