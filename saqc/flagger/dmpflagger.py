#! /usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import pandas as pd

from .baseflagger import BaseFlagger


class Keywords:
    VERSION = "$version"


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
        self.flag_fields = [FlagFields.FLAG,
                            FlagFields.CAUSE,
                            FlagFields.COMMENT]
        version = subprocess.check_output(
            'git describe --tags --always --dirty'.split())
        self.project_version = version.decode().strip()

    def initFlags(self, data, **kwargs):
        if isinstance(data, pd.Series):
            data = data.to_frame()

        colindex = pd.MultiIndex.from_product(
            [data.columns, self.flag_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])

        out = pd.DataFrame(data=self.flags[0],
                           columns=colindex,
                           index=data.index)
        return out.astype(
            {c: self.flags for c in out.columns if FlagFields.FLAG in c})

    def setFlag(self, flags, flag=None, cause="", comment="", **kwargs):

        if not isinstance(flags, pd.DataFrame):
            raise TypeError

        flag = self.BAD if flag is None else self._checkFlag(flag)

        if Keywords.VERSION in comment:
            comment = comment.replace(Keywords.VERSION, self.project_version)

        flags = self._reduceColumns(flags)
        mask = flags[FlagFields.FLAG] < flag
        flags.loc[mask, self.flag_fields] = flag, cause, comment

        return flags.values

    def isFlagged(self, flags, flag=None, comparator=">"):
        flags = self._reduceColumns(flags)
        flagcol = flags.loc[:, FlagFields.FLAG].squeeze()
        return super().isFlagged(flagcol, flag, comparator)

    def _reduceColumns(self, flags):
        if set(flags.columns) == set(self.flag_fields):
            pass
        elif isinstance(flags, pd.DataFrame) \
                and isinstance(flags.columns, pd.MultiIndex) \
                and (len(flags.columns) == 3):
            flags = flags.copy()
            flags.columns = flags.columns.get_level_values(ColumnLevels.FLAGS)
        else:
            raise TypeError
        return flags
