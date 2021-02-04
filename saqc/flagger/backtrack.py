#!/usr/bin/env python
from __future__ import annotations

from typing import Type, Any, Tuple
import pandas as pd
import numpy as np
import dios

# fixme: if we decide for a container
#  check every occurrence of this


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
            self._validate_bt(bt, mask)

        if copy:
            bt = bt.copy()
            mask = mask.copy()

        self.bt = bt
        self.mask = mask
        self._nr = len(bt)

    @property
    def nr(self):
        return self._nr

    def next(self):
        self._nr += 1
        return self

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
        self.next()
        return self

    def max(self):
        """
        Get the maximum value per row of non-masked data.

        Returns
        -------
        pd.Series: maximum values
        """
        return self.bt[self.mask].max(axis=1)


