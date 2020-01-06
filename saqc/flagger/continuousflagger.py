#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import intervals

from saqc.flagger.baseflagger import BaseFlagger


class ContinuousFlagger(BaseFlagger):
    def __init__(self, min_=0.0, max_=1.0, unflagged=-1.0):
        assert unflagged < 0 <= min_ < max_
        super().__init__(dtype=np.float64)
        self._interval = intervals.closed(min_, max_)
        self._unflagged_flag = unflagged
        self.signature = ("flag", "factor", "modify")

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, factor=1, modify=False, **kwargs):
        # NOTE: incomplete, as the option to
        #       update flags is temporarily gone
        return super().setFlags(field=field, loc=loc, iloc=iloc, flag=self._checkFlag(flag), force=force, **kwargs)

    # NOTE:
    # we should probably override _assureDtype here

    def _isDtype(self, flag):
        if isinstance(flag, pd.Series):
            # NOTE: it should be made sure, that all
            #       values fall into the interval
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
