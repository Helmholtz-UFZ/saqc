#! /usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import json
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
        version = subprocess.run(
            "git describe --tags --always --dirty",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
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

        # if Keywords.VERSION in comment:
        comment = json.dumps({
            "comment": comment,
            "commit": self.project_version,
            "test": kwargs.get("func_name", "")})

        flags = self._reduceColumns(flags)
        mask = flags[FlagFields.FLAG] < flag
        if isinstance(flag, pd.Series):
            flags.loc[mask, self.flag_fields] = flag[mask], cause, comment
        else:
            flags.loc[mask, self.flag_fields] = flag, cause, comment
        return flags.values

    def isFlagged(self, flags, flag=None, comparator=">"):
        flagcol = self.getFlags(flags)
        return super().isFlagged(flagcol, flag, comparator)

    def getFlags(self, flags):
        if isinstance(flags, pd.Series):
            return super().getFlags(flags)

        elif isinstance(flags, pd.DataFrame):
            if isinstance(flags.columns, pd.MultiIndex):
                f = flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
            else:
                f = flags.loc[:, FlagFields.FLAG]
        else:
            raise TypeError(flags)
        return f.squeeze()

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
