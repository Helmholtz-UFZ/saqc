#! /usr/bin/env python
# -*- coding: utf-8 -*-

import operator as op
from typing import Any, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

from .baseflagger import BaseFlagger
from ..lib.types import PandasLike, ArrayLike, T
from saqc.lib.tools import check_isdf
from pandas.api.types import CategoricalDtype


class Flags(pd.CategoricalDtype):
    def __init__(self, flags):
        # NOTE: all flag schemes need to support
        #       at least 3 flag categories:
        #       * unflagged
        #       * good
        #       * bad
        assert len(flags) > 2
        super().__init__(flags, ordered=True)


class CategoricalBaseFlagger(BaseFlagger):

    def __init__(self, flags):
        super().__init__(dtype=Flags(flags))
        self._categories = self.dtype.categories
        self.signature = ("flag", "force")

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        self._flags = self._assureDtype(flags)
        return self

    def _checkFlag(self, flag):
        if not self._isCategorical(flag):
            raise TypeError(
                f"invalid flag '{flag}', possible choices are '{list(self.categories.categories)}'")
        return flag

    def _assureDtype(self, flags):
        # NOTE: building up new DataFrames is significantly
        #       faster than assigning into existing ones
        if isinstance(flags, pd.Series):
            return flags.astype(self.dtype)
        tmp = OrderedDict()
        for c in flags.columns:
            tmp[c] = flags[c].astype(self.dtype)
        return pd.DataFrame(tmp)

    def _checkFlag(self, flag):
        if not self._isCategorical(flag):
            raise TypeError(
                f"invalid flag '{flag}', possible choices are '{list(self.dtype.categories)}'")
        return flag

    def nextTest(self):
        pass

    @property
    def UNFLAGGED(self):
        return self._categories[0]

    @property
    def GOOD(self):
        return self._categories[1]

    @property
    def BAD(self):
        return self._categories[-1]

    def isSUSPICIOUS(self, flag):
        return flag in self._categories[2:-1]
