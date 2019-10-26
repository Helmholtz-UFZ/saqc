#! /usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import json
import pandas as pd

from .baseflagger import BaseFlagger, PandasLike


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
        # check
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be of type pd.DataFrame, {type(data)} was given")
        # create
        colindex = pd.MultiIndex.from_product(
            [data.columns, self.flag_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])
        flags = pd.DataFrame(data=self.flags[0], columns=colindex, index=data.index)
        return self._assureDtype(flags)

    def isFlagged(self, flags: PandasLike, flag=None, comparator=">") -> PandasLike:
        flags = self.getFlags(flags)
        return super().isFlagged(flags, flag, comparator)

    def getFlags(self, flags: PandasLike) -> PandasLike:
        if not isinstance(flags, pd.DataFrame):
            super().getFlags(flags)
        if isinstance(flags.columns, pd.MultiIndex):
            flags = flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
        else:
            flags = flags[FlagFields.FLAG].squeeze()
        return flags

    def setFlags(self, flags, field, mask_or_indexer=None, flag=None, comment='', cause='', **kwargs):
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if not isinstance(flags.columns, pd.MultiIndex):
            raise TypeError(f"flags.index must be a multiindex")
        # prepare
        ysoncomment = json.dumps(dict(comment=comment, commit=self.project_version, test=kwargs.get("func_name", "")))
        comment = '' if comment is None else ysoncomment
        flags = self._assureDtype(flags, field)
        moi = slice(None) if mask_or_indexer is None else mask_or_indexer
        flag = self.BAD if flag is None else self._checkFlag(flag)
        # set
        f = self.getFlags(flags[field])
        if isinstance(flag, pd.Series):
            flag = flag.loc[moi]
        mask = f.loc[moi] < flag
        idx = mask[mask].index
        flags.loc[idx, field] = flag, cause, comment
        return self._assureDtype(flags, field)

    def clearFlags(self, flags, field, mask_or_indexer=None, **kwargs):
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if not isinstance(flags.columns, pd.MultiIndex):
            raise TypeError(f"flags.index must be a multiindex")
        # prepare
        flags = self._assureDtype(flags, field)
        moi = slice(None) if mask_or_indexer is None else mask_or_indexer
        # set
        flags.loc[moi, field] = self.UNFLAGGED, '', ''
        return self._assureDtype(flags, field)

    def _assureDtype(self, flags, field=None):
        if field is None:
            if isinstance(flags, pd.Series):
                flags = super()._assureDtype(flags, None)
            else:
                flags = flags.astype({c: self.flags for c in flags.columns if FlagFields.FLAG in c})
        elif not isinstance(flags[(field, FlagFields.FLAG)].dtype, pd.Categorical):
            flags[(field, FlagFields.FLAG)] = flags[(field, FlagFields.FLAG)].astype(self.flags)
        return flags
