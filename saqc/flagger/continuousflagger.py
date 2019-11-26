#! /usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import pandas as pd
import numpy as np
import intervals

from saqc.flagger.baseflagger import BaseFlagger
from saqc.lib.tools import isDataFrameCheck


class ContinuousBaseFlagger(BaseFlagger):

    def __init__(self, min_=0., max_=1., unflagged=-1.):
        assert unflagged < 0 <= min_ < max_
        super().__init__(dtype=np.float64)
        self._interval = intervals.closed(min_, max_)
        self._unflagged_flag = unflagged
        self.signature = ("flag", "factor", "modify")

    def initFlags(self, data: pd.DataFrame = None, flags: pd.DataFrame = None):
        if data is not None:
            isDataFrameCheck(data, 'data', allow_multiindex=False)
            flags = pd.DataFrame(
                data=self.UNFLAGGED, index=data.index, columns=data.columns)
        elif flags is not None:
            isDataFrameCheck(flags, 'flags', allow_multiindex=False)
        out = deepcopy(self)
        out._flags = flags
        return out

    def isFlagged(self, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        return super().isFlagged(
            field=field, loc=loc, iloc=iloc,
            flag=self._checkFlag(flag), comparator=comparator,
            **kwargs)

    def setFlags(self, field, loc=None, iloc=None, flag=None,
                 force=False, factor=1, modify=False, **kwargs):
        # NOTE: incomplete, as the option to
        #       update flags is temporarily gone
        return super().setFlags(
            field=field, loc=loc, iloc=iloc,
            flag=self._checkFlag(flag), force=force,
            **kwargs)

    def _checkFlag(self, flag):
        if flag is not None and not self._isInterval(flag):
            raise TypeError(
                f"invalid flag: '{flag}'")
        return flag

    def _isInterval(self, flag):
        if isinstance(flag, pd.Series):
            return flag.dtype == self.dtype
        return flag in self._interval or flag == self.UNFLAGGED

    @property
    def UNFLAGGED(self):
        return self._unflagged_flag

    @property
    def GOOD(self):
        return self._interval.lower

    @property
    def BAD(self):
        return self._interval.upper

    def isSUSPICIOUS(self, flag):
        return flag in intervals.open(self.GOOD, self.BAD)
