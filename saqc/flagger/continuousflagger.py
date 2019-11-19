#! /usr/bin/env python
# -*- coding: utf-8 -*-


from saqc.flagger.template import FlaggerTemplate
from saqc.flagger.categoricalflagger import CategoricalFlagger
from saqc.lib.tools import check_isdf
import pandas as pd
import numpy as np
import intervals as I

class ContinuousFlagger(FlaggerTemplate):

    def __init__(self, min_=0., max_=1., unflagged=-1.):
        assert unflagged < 0 <= min_ < max_
        super().__init__(np.dtype(float))
        self._interval = I.closed(min_, max_)
        self._unflagged_flag = unflagged

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        return flags

    def _isDtype(self, f) -> bool:
        return f in self._interval or f < 0

    def _checkFlag(self, f):
        if not self._isDtype(f):
            raise TypeError(f'flag must be in [{self._interval.lower}, {self._interval.upper}], {f} was passed')
        return f

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, factor=1, modify=False, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')

        flags = flags.copy()

        if modify is True:
            values = self.getFlags(flags, field, loc, iloc, **kwargs)
            idx = values.index
        else:
            flag = self.BAD if flag is None else self._checkFlag(flag)
            idx, values = self._prepareWrite(flags, field, flag, loc, iloc, force, **kwargs)

        values *= factor
        self._writeFlags(flags, field, idx, values, **kwargs)
        return flags

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
        return flag in I.open(self.GOOD, self.BAD)


if __name__ == '__main__':
    flagger = ContinuousFlagger()
    a = 'a'
    df = pd.DataFrame(dict(a=[1,2,3,4]))
    flags = flagger.initFlags(df)
    flags = flagger.getFlags(flags)

    print(flagger.dtype)
    print(flags[a].dtype)

    # flags = flags.astype(flagger.dtype)
    # print(flagger.dtype)
    # print(flags[a].dtype)

    for dt in flags.dtypes:
        print(dt, type(flagger.dtype))
        assert isinstance(dt, type(flagger.dtype))
