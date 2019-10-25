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
        self.flag_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]
        version = subprocess.run("git describe --tags --always --dirty",
                                 shell=True, check=False, stdout=subprocess.PIPE).stdout
        self.project_version = version.decode().strip()

    def initFlags(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be of type pd.DataFrame, {type(data)} was given")
        colindex = pd.MultiIndex.from_product(
            [data.columns, self.flag_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])
        flags = pd.DataFrame(data=self.flags[0], columns=colindex, index=data.index)
        flags = flags.astype({c: self.flags for c in flags.columns if FlagFields.FLAG in c})
        return flags

    def isFlagged(self, flags: pd.DataFrame, flag=None, comparator=">") -> pd.DataFrame:
        flags = self.getFlags(flags)
        return super().isFlagged(flags, flag, comparator)

    def getFlags(self, flags: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if isinstance(flags.columns, pd.MultiIndex):
            flags = flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
        else:
            flags = flags[FlagFields.FLAG].squeeze()
        return flags

    def setFlags(self, flags, field, mask_or_indexer=None, flag=None, comment='', cause='', **kwargs):
        comment = json.dumps({
            "comment": comment,
            "commit": self.project_version,
            "test": kwargs.get("func_name", "")})

        flags = flags.copy()
        r = slice(None) if mask_or_indexer is None else mask_or_indexer
        flag = self.BAD if flag is None else self._checkFlag(flag)
        f = self.getFlags(flags[field])
        mask = f.loc[r] < flag
        idx = mask[mask].index
        flags.loc[idx, field] = flag, cause, comment
        return flags

    def setFlag(self, flags, flag=None, cause="", comment="", **kwargs):
        flags = self._checkFlagsType(flags)
        flag = self.BAD if flag is None else self._checkFlag(flag)

        # if Keywords.VERSION in comment:
        comment = json.dumps({
            "comment": comment,
            "commit": self.project_version,
            "test": kwargs.get("func_name", "")})

        mask = flags[FlagFields.FLAG] < flag
        if isinstance(flag, pd.Series):
            flags.loc[mask, self.flag_fields] = flag[mask], cause, comment
        else:
            flags.loc[mask, self.flag_fields] = flag, cause, comment
        return flags.values

    def clearFlags(self, flags, **kwargs):
        flags = self._checkFlagsType(flags)
        flags.loc[:, self.flag_fields] = self.UNFLAGGED, "", ""
        return flags.values

    def _checkFlagsType(self, flags):
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if set(flags.columns) != set(self.flag_fields):
            colstr = f"{list(flags.columns)[0:4]} ..." if len(flags.columns) > 4 else f"{list(flags.columns)[0:4]}"
            raise TypeError(f"flags must have the exact columns: {self.flag_fields}, but {colstr} was given")
        return flags
