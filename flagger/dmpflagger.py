#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .baseflagger import BaseFlagger

class FlagFields:
    FLAG = "quality_flag"
    CAUSE = "quality_cause"
    COMMENT = "quality_comment"


class Flags:
    OK = "OK"
    DOUBTFUL = "DOUBTFUL"
    BAD = "BAD"

    @staticmethod
    def isValid(flag):
        return flag in [Flags.OK, Flags.DOUBTFUL, Flags.BAD]


class DmpFlagger(BaseFlagger):
    def __init__(self, no_flag="NIL", flag="BAD"):
        super().__init__(no_flag, flag)
        self.flag_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]

    def emptyFlags(self, data, value="NIL", **kwargs):
        columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]
        columns = pd.MultiIndex.from_product([columns, self.flag_fields])
        return pd.DataFrame(data=value, columns=columns, index=data.index)

    def setFlag(self, flags, flag=Flags.BAD, cause="NIL", comment="NIL", **kwargs):
        self._isFlag(flag)
        for field, f in zip(self.flag_fields, [flag, cause, comment]):
            flags.loc[:, field] = f
        return flags

    def isFlagged(self, flags, flag=None):
        flagcol = flags.loc[:, FlagFields.FLAG].squeeze()
        return super().isFlagged(flagcol, flag)

    def _isFlag(self, flag):
        assert Flags.isValid(flag)

