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


class FlaggerTemplate(ABC):
    """
    General implementation constrains for all public methods:
     - the `flags` input must be of same type and dimensions of whatever self.initFlags returns
     - these original `flags` must not be modified. use flags.copy() !
     - `loc`, `iloc` should behave the same in all public methods, and do the same as described in [1]
     - implemt the loc/iloc-behavior in self._reduceRows()
     - if `flag` is None it should default to any of the flags(categories) from __init__, normally
       this defaults to the possibly best or worst flag

    Notes: [1]
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html,
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
    """

    @abstractmethod
    def __init__(self, flags):
        ...

    @abstractmethod
    def initFlags(self, data: pd.DataFrame) -> newT:
        ...

    @abstractmethod
    def isFlagged(self, flags: newT, field: str = None, loc=None, iloc=None, flag=None,
                  comparator: str = ">", **kwargs) -> PandasLike:
        """
        Return bool information on flags.

        Implementation constrains:
         - return a pandas.DataFrame, if `field` is None
         - return a pandas.Series, otherwise.
         - the `comparator` can be used to alter the behavior of the comparison between `flags` and `flag`

        calls:
         - self.getFlags() or all subcalls there
         - self._checkFlag()
        """
        # NOTE: I dislike the comparator default, as it does not comply with
        #       the setFlag defautl behaviour, which is not changable, btw
        ...

    @abstractmethod
    def getFlags(self, flags: newT, field: str = None, loc=None, iloc=None, **kwargs):
        """
        Return the flags information, reduced to only the pure flag information

        Implementation constrains:
         - return a pandas.DataFrame, if `field` is None
         - return a pandas.Series, otherwise.

        calls:
         - self._checkFlags()
         - self._reduceColumns()
         - self._reduceRows()
        """
        ...

    @abstractmethod
    def setFlags(self, flags: newT, field: str, loc=None, iloc=None, flag=None, force=False, **kwargs) -> newT:
        """
        Set flags, if flags are lower in order than flag.

        Implementation constrains:
         - return the a modified copy of flags with same dimension than the original
         - this should call self._writeFlags()

        calls:
         - self.getFlags() or all subcalls there
         - self._writeFlags()
         - self._checkFlag()
        """
        ...

    @abstractmethod
    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        """
        Clear flags.

        Implementation constrains:
         - return the a modified copy of flags with same dimension than the original

        calls:
         - self.getFlags() or all subcalls there
         - self._writeFlags()
        """
        ...

    @abstractmethod
    def _reduceColumns(self, df: newT, field=None, **kwargs) -> pd.DataFrame:
        """ Reduce a object of your desired type to a simple single-indexed pandas.DataFrame """
        ...

    @abstractmethod
    def _reduceRows(self, df_or_ser: PandasLike, field, loc, iloc, **kwargs) -> PandasLike:
        """ Reduce rows with the given loc or iloc. May also reduce a df to a series"""
        ...

    @abstractmethod
    def _writeFlags(self, flags, rowindex, field, flag, **kwargs):
        """ Write unconditional(!) to flags """
        ...

    @abstractmethod
    def _checkFlags(self, flags, **kwargs):
        """ Check if the flags input frame is valid """
        ...

    @abstractmethod
    def _checkFlag(self, flag, **kwargs):
        """ Check if the flag parmeter is valid """
        ...

    @property
    @abstractmethod
    def UNFLAGGED(self):
        """ Return the category that indicates that the flag is unflagged"""
        ...

    @property
    @abstractmethod
    def GOOD(self):
        """ Return the category that indicates the best flag"""
        ...

    @property
    @abstractmethod
    def BAD(self):
        """ Return the category that indicates the worst flag"""
        ...

    @property
    @abstractmethod
    def SUSPICIOUS(self):
        """ Return the categories that lie between self.GOOD and self.BAD"""
        ...
