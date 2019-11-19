#! /usr/bin/env python
# -*- coding: utf-8 -*-


from saqc.flagger.template import FlaggerTemplate
from saqc.flagger.baseflagger import BaseFlagger
from saqc.lib.tools import check_isdf
import pandas as pd
import intervals as I

FLAGS = [-1, 0, 1]


class PercentageFlagger(FlaggerTemplate):

    def __init__(self):
        super().__init__(float)

    def initFlags(self, data: pd.DataFrame):
        check_isdf(data, 'data', allow_multiindex=False)
        flags = pd.DataFrame(data=self.UNFLAGGED, index=data.index, columns=data.columns)
        return flags

    def _assureDtype(self, flags, field=None, **kwargs):
        return flags

    def _isSelfCategoricalType(self, f) -> bool:
        if isinstance(f, float):
            return self.UNFLAGGED() <= f <= self.GOOD()
        return False

    def setFlags(self, *args, factor=None, **kwargs):
        return super().setFlags(*args, factor=factor, **kwargs)

    def _writeFlags(self, flags, rowindex, field, flag, factor=None, **kwargs):
        flags.loc[rowindex, field] = flag
            if factor is not None:

        return flags

    def UNFLAGGED(self):
        return -1.

    def GOOD(self):
        return 0.

    def SUSPICIOUS(self):
        return I.open(0., 1.)

    def BAD(self):
        return 1.


if __name__ == '__main__':
    f = PercentageFlagger()
