#! /usr/bin/env python
# -*- coding: utf-8 -*-


from .template import FlaggerTemplate
from ..lib.tools import check_isdf
import pandas as pd
import numpy as np
import intervals as I


class ContinuousFlagger(FlaggerTemplate):

    def __init__(self, min_=0., max_=1., unflagged=-1.):
        assert unflagged < 0 <= min_ < max_
        super().__init__(dtype=np.float64)
        self._interval = I.closed(min_, max_)
        self._unflagged_flag = unflagged

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        return flags

    def isFlagged(self, flags, field=None, loc=None, iloc=None, flag=None, comparator: str = ">", **kwargs):
        flag = self.GOOD if flag is None else self._checkFlag(flag, allow_series=False)
        return super().isFlagged(flags, field, loc, iloc, flag, comparator, **kwargs)

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, factor=1, modify=False, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        flags = flags.copy()

        if modify is False:
            # normal setFlags
            flag = self.BAD if flag is None else self._checkFlag(flag, allow_series=True)
            idx, values = self._prepareWrite(flags, field, flag, loc, iloc, force, **kwargs)
            self._writeFlags(flags, field, idx, values, **kwargs)

        return self._modFlags(flags, field, loc, iloc, factor)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        return super().clearFlags(flags, field, loc, iloc, **kwargs)

    def _writeFlags(self, flags, field, rowindex, values, **kwargs):
        flags.loc[rowindex, field] = values
        return flags

    def _modFlags(self, flags, field, loc, iloc, factor):
        flagged = self.isFlagged(flags, field, loc, iloc, flag=self.UNFLAGGED)
        flags.loc[flagged.index] *= factor
        return flags

    def _checkFlag(self, f, allow_series=False):
        if f is None:
            raise ValueError('flag cannot be None')

        if allow_series is False and hasattr(f, '__len__'):
            raise ValueError('flag must be a single value here')

        if isinstance(f, pd.Series):
            if f.dtype != self.dtype:
                raise TypeError(f'flag series dtype must be {self.dtype}')

        elif f not in self._interval and f != self._unflagged_flag:
            raise TypeError(f'flag must be in [{self._interval.lower}, {self._interval.upper}] or '
                            f'{self._unflagged_flag} (unflagged), {f} was passed')
        return f

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
