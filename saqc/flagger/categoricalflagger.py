#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from copy import deepcopy

import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.lib.tools import isDataFrameCheck


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

    def initFlags(self, data=None, flags=None):
        if data is not None:
            return self._initFromData(data)
        elif flags is not None:
            return self._initFromFlags(flags)
        else:
            raise TypeError("either 'data' or 'flags' are required")

    def _initFromData(self, data: pd.DataFrame):
        isDataFrameCheck(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(
            data=self.UNFLAGGED, index=data.index, columns=data.columns)
        self._flags = self._assureDtype(flags)
        return self

    def _initFromFlags(self, flags: pd.DataFrame):
        isDataFrameCheck(flags, 'flags', allow_multiindex=False)
        out = deepcopy(self)
        out._flags = out._assureDtype(flags)
        return out

    def getFlags(self, field=None, loc=None, iloc=None, **kwargs):
        flags = super().getFlags(field, loc, iloc)
        return self._assureDtype(flags)

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, **kwargs):
        return super().setFlags(
            field=field, loc=loc, iloc=iloc,
            flag=self._checkFlag(flag), force=force,
            **kwargs)

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        return super().isFlagged(
            field=field, loc=loc, iloc=iloc,
            flag=self._checkFlag(flag), comparator=comparator,
            **kwargs)

    def _assureDtype(self, flags):
        # NOTE: building up new DataFrames is significantly
        #       faster than assigning into existing ones
        if isinstance(flags, pd.Series):
            return flags.astype(self.dtype)
        tmp = OrderedDict()
        for c in flags.columns:
            tmp[c] = flags[c].astype(self.dtype)
        return pd.DataFrame(tmp)

    def _isCategorical(self, f) -> bool:
        """
        not needed here, move out
        """
        if isinstance(f, pd.Series):
            return isinstance(f.dtype, pd.CategoricalDtype) and f.dtype == self.dtype
        return f in self.dtype.categories

    def _checkFlag(self, flag):
        if flag is not None and not self._isCategorical(flag):
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
