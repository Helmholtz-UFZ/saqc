#! /usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from .baseflagger import BaseFlagger


class FlagFields:
    FLAG = "quality_flag"
    CAUSE = "quality_cause"
    COMMENT = "quality_comment"


class ColumnLevels:
    VARIABLES = "variables"
    FLAGS = "flags"


FLAGS = ["NIL", "OK", "DOUBTFUL", "BAD"]


class DmpFlagger(BaseFlagger):

    def __init__(self):
        super().__init__(FLAGS)
        self.flag_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]

    def initFlags(self, data, **kwargs):
        columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]

        colindex = pd.MultiIndex.from_product(
            [columns, self.flag_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])

        out = pd.DataFrame(data=self.flags[0],
                           columns=colindex,
                           index=data.index)
        return out.astype(
            {c: self.flags for c in out.columns if FlagFields.FLAG in c})

    def setFlag(self, flags, flag=None, cause="", comment="", **kwargs):

        if flag is None:
            flag = self.flags.max()
        assert flag in self.flags

        flags = self._reduceColumns(flags)
        flags.loc[flags[FlagFields.FLAG] < flag, FlagFields.FLAG] = flag
        for field, f in [(FlagFields.CAUSE, cause), (FlagFields.COMMENT, comment)]:
            flags.loc[:, field] = f

        return flags.values

    def isFlagged(self, flags, flag=None):
        flags = self._reduceColumns(flags)
        flagcol = flags.loc[:, FlagFields.FLAG].squeeze()
        return super().isFlagged(flagcol, flag)

    def _reduceColumns(self, flags):
        if isinstance(flags.columns, pd.MultiIndex):
            flags.columns = flags.columns.get_level_values(ColumnLevels.FLAGS)
        return flags

    # def _isFlag(self, flag):
    #     assert Flags.isValid(flag)
