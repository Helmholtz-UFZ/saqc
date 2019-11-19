#! /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import operator as op
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..lib.types import PandasLike, ArrayLike, T
from ..lib.tools import *
from typing import TypeVar

newT = TypeVar("newT")

COMPARATOR_MAP = {
    "==": op.eq,
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
}


class FlaggerTemplate(ABC):
    """
    General implementation constrains for all public methods:
     - the `flags` input must be of same type and dimensions of whatever self.initFlags returns
     - these original `flags` must not be modified. use flags.copy() !
     - `loc`, `iloc` should behave the same in all public methods, and do the same as described in [1]
     - implemt the loc/iloc-behavior in self._reduceRows()
     - all additional instance attributes (that are not present in *all* flaggers) should start single
     _underscored (private) or even __double_underscored.

    Notes: [1]
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html,
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
    """

    @abstractmethod
    def __init__(self, flags_dtype):
        """ Init the class and set the categories (in param flags) that are used here."""
        self.dtype = flags_dtype

    def initFlags(self, data: pd.DataFrame) -> newT:
        """ Prepare the flags to your desire. data is passed as reference shape """
        ...

    def isFlagged(self, flags: newT, field: str = None, loc=None, iloc=None, flag=None,
                  comparator: str = ">", **kwargs) -> PandasLike:
        """
        Return bool information on flags.

        Implementation constrains:
         - return a pandas.DataFrame, if `field` is None
         - return a pandas.Series, otherwise.
         - the `comparator` can be used to alter the behavior of the comparison between `flags` and `flag`
        """
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        flags = self.getFlags(flags, field, loc, iloc, **kwargs)
        cp = COMPARATOR_MAP[comparator]
        flagged = pd.notna(flags) & cp(flags, flag)
        return flagged

    def getFlags(self, flags: newT, field: str = None, loc=None, iloc=None, **kwargs):
        """
        Return the flags information, reduced to only the pure flag information

        Implementation constrains:
         - return a pandas.DataFrame, if `field` is None
         - return a pandas.Series, otherwise.
        """
        flags = flags.copy()
        flags = self._reduceColumns(flags, field, **kwargs)
        flags = self._reduceRows(flags, loc, iloc, **kwargs)
        return flags

    def setFlags(self, flags: newT, field: str, loc=None, iloc=None, flag=None, force=False, **kwargs) -> newT:
        """
        Set flags, if flags are lower in order than flag.

        Implementation constrains:
         - return the a modified copy of flags with same dimension than the original
        """
        flags = flags.copy()
        idx, values = self._prepareWrite(flags, field, flag, loc, iloc, force, **kwargs)
        flags = self._writeFlags(flags, field, idx, values, **kwargs)
        return flags

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        """
        Clear flags.

        Implementation constrains:
         - return the a modified copy of flags with same dimension than the original
        """
        flags = flags.copy()
        kwargs.pop('force', None)
        idx, values = self._prepareWrite(flags, field, self.UNFLAGGED, loc, iloc, force=True, **kwargs)
        flags = self._writeFlags(flags, field, idx, values, **kwargs)
        return flags

    def _reduceColumns(self, df: newT, field, **kwargs) -> pd.DataFrame:
        """ Reduce a object of your desired type and return
         a pd.DataFrame [1] if field is None, otherwise a pd.Series

        [1] a standard `normal`-indexed pandas.DataFrame
        """
        return df if field is None else df[field]

    def _reduceRows(self, df_or_ser: PandasLike, loc, iloc, **kwargs) -> PandasLike:
        """ Reduce rows with the given loc or iloc """
        if loc is not None and iloc is not None:
            raise ValueError("params `loc` and `iloc` are mutual exclusive")
        elif loc is not None and iloc is None:
            return df_or_ser.loc[loc]
        elif loc is None and iloc is not None:
            return df_or_ser.iloc[iloc]
        elif loc is None and iloc is None:
            return df_or_ser

    def _prepareWrite(self, dest, field, src, loc, iloc, force, **kwargs):
        dest = dest.copy()
        dest = self._reduceColumns(dest, field, **kwargs)

        if not isinstance(src, pd.Series):
            # assert len(src) == len(dest) if hasattr(src, '__len__') else True
            src = pd.Series(data=src, index=dest.index).astype(self.dtype)

        # shrink data
        dest = self._reduceRows(dest, loc, iloc, **kwargs)
        src = self._reduceRows(src, loc, iloc, **kwargs)

        # assert isinstance(dest, pd.Series) and isinstance(src, pd.Series) and len(src) == len(dest)
        if force:
            idx = dest.index
        else:
            mask = dest < src
            idx = dest[mask].index
            src = src[mask]
        return idx, src

    def _writeFlags(self, flags, field, rowindex, values, **kwargs):
        """ Write unconditional(!) to flags """
        flags.loc[rowindex, field] = values
        return flags

    @property
    @abstractmethod
    def UNFLAGGED(self):
        """ Return the flag that indicates unflagged data """
        ...

    @property
    @abstractmethod
    def GOOD(self):
        """ Return the flag that indicates the very best data """
        ...

    @property
    @abstractmethod
    def BAD(self):
        """ Return the flag that indicates the worst data """
        ...

    @abstractmethod
    def isSUSPICIOUS(self, flag):
        """ Return bool that indicates if the given flag is valid, but neither
        UNFLAGGED, BAD, nor GOOD."""
        ...
