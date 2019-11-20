#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional

import numpy as np
import pandas as pd

from .template import FlaggerTemplate
from ..lib.types import PandasLike, ArrayLike, T
from ..lib.tools import *
from pandas.api.types import CategoricalDtype


class Flags(CategoricalDtype):

    def put(self, *args, **kwargs):
        raise NotImplementedError('modification not implemented')

    def __init__(self, flags):
        # NOTE: all flag schemes need to support
        #       at least 3 flag categories:
        #       * unflagged
        #       * good
        #       * bad
        assert len(flags) > 2
        super().__init__(flags, ordered=True)

    def unflagged(self):
        return self[0]

    def good(self):
        return self[1]

    def bad(self):
        return self[-1]

    def suspicious(self):
        return self[2:-1]

    def __getitem__(self, idx):
        return self.categories[idx]


class CategoricalFlagger(FlaggerTemplate):
    def __init__(self, flags):
        self.signature = ("flag", "force")
        self.__categories = Flags(flags)
        super().__init__(flags_dtype=self.__categories)

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        return self._assureDtype(flags)

    def isFlagged(self, flags, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        flag = self.GOOD if flag is None else self._checkFlag(flag)
        flagged = super().isFlagged(flags, field, loc, iloc, flag, comparator, **kwargs)
        return flagged

    def getFlags(self, flags, field=None, loc=None, iloc=None, **kwargs):
        flags = self._checkFlags(flags, **kwargs)
        flags = super().getFlags(flags, field, loc, iloc)
        return self._assureDtype(flags, field, **kwargs)

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        flag = self.BAD if flag is None else self._checkFlag(flag, allow_series=True, lenght=len(flags))
        flags = self._checkFlags(flags, **kwargs)
        flags = super().setFlags(flags, field, loc, iloc, flag, force, **kwargs)
        return self._assureDtype(flags, field, **kwargs)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        flags = super().clearFlags(flags, field, loc, iloc, **kwargs)
        return self._assureDtype(flags)

    def _reduceRows(self, df_or_ser, loc, iloc, **kwargs) -> pd.DataFrame:
        prelen = len(df_or_ser)
        df_or_ser = super()._reduceRows(df_or_ser, loc, iloc)
        if prelen == len(df_or_ser):
            # this is a cheap OPTIMISATION, and works because we only
            # loose the dtype if all values was overwritten
            df_or_ser = self._assureDtype(df_or_ser)
        return df_or_ser

    def _checkFlags(self, flags, **kwargs):
        check_isdf(flags, argname='flags')
        return flags

    def _checkFlag(self, flag, allow_series=False, lenght=None):
        if flag is None:
            raise ValueError("flag cannot be None")

        if isinstance(flag, pd.Series):
            if not allow_series:
                raise TypeError('series of flags are not allowed here')

            if not isinstance(flag.dtype, self.dtype):
                raise TypeError(f"flag(-series) is not of expected '{self.dtype}'-dtype with ordered categories "
                                f"{list(self.__categories.categories)}, '{flag.dtype}'-dtype was passed.")

            assert lenght is not None, 'faulty Implementation, length param must be given if flag is a series'
            if len(flag) != lenght:
                raise ValueError(f'length of flags ({lenght}) and flag ({len(flag)}) must match, if flag is '
                                 f'a series')

        elif flag not in self.__categories:
            raise TypeError(f"Invalid flag '{flag}'. Possible choices are {list(self.__categories.categories)}")

        return flag

    def _assureDtype(self, flags, field=None, **kwargs):
        # in: df/ser, out: df/ser, affect only the minimal set of columns
        if isinstance(flags, pd.Series):
            return flags if isinstance(flags.dtype, self.dtype) else flags.astype(self.dtype)

        elif isinstance(flags, pd.DataFrame):
            if field is None:
                for c in flags:
                    flags[c] = self._assureDtype(flags[c], **kwargs)
            else:
                flags[field] = self._assureDtype(flags[field], **kwargs)
        else:
            raise NotImplementedError

        return flags

    def nextTest(self):
        pass

    @property
    def UNFLAGGED(self):
        return self.__categories.unflagged()

    @property
    def GOOD(self):
        return self.__categories.good()

    @property
    def BAD(self):
        return self.__categories.bad()

    def isSUSPICIOUS(self, flag):
        return flag in self.__categories.suspicious()
