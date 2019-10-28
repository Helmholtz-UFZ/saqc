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
        if not isinstance(flags, pd.DataFrame):  # series
            super().getFlags(flags)
        elif isinstance(flags.columns, pd.MultiIndex):  # df, multiindex
            flags = flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
        else:  # df, simple index
            # if we come here a dataframe with multiindex, was unpacked by a fieldname, like so: getFlags(df[field])
            # so we can be sure to have qflag, qcause, qcomment as df-columns (otherwise something went wrong on the
            # user side). As so we need to squeeze the result to a series, because the df wont have any column info,
            # like to which field/variable the qflags belongs to.
            flags = flags[FlagFields.FLAG].squeeze()
        return flags

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, comment='', cause='', **kwargs):
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if not isinstance(flags.columns, pd.MultiIndex):
            raise TypeError(f"flags.index must be a multiindex")
        # prepare
        ysoncomment = json.dumps(dict(comment=comment, commit=self.project_version, test=kwargs.get("func_name", "")))
        comment = '' if comment is None else ysoncomment
        flags = self._assureDtype(flags, field)
        flag = self.BAD if flag is None else self._checkFlag(flag)
        # set
        indexer, rows, col = self._getIndexer(self.getFlags(flags), field, loc, iloc)
        if isinstance(flag, pd.Series):
            i, r, _ = self._getIndexer(flag, field, loc, iloc)
            flag = i[r]
        mask = indexer[rows, col] < flag
        idx = mask[mask].index
        flags.loc[idx, field] = flag, cause, comment
        return self._assureDtype(flags, field)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if not isinstance(flags, pd.DataFrame):
            raise TypeError(f"flags must be of type pd.DataFrame, {type(flags)} was given")
        if not isinstance(flags.columns, pd.MultiIndex):
            raise TypeError(f"flags.index must be a multiindex")
        indexer, rows, col = self._getIndexer(flags, field, loc, iloc)
        indexer[rows, col] = self.UNFLAGGED, '', ''
        return self._assureDtype(flags, field)

    def _assureDtype(self, flags, field=None):
        if field is None:
            if isinstance(flags, pd.Series):  # we got a series
                flags = super()._assureDtype(flags, None)
            else:  # we got a df with multiindex
                flags = flags.astype({c: self.flags for c in flags.columns if FlagFields.FLAG in c})
        elif not isinstance(flags[(field, FlagFields.FLAG)].dtype, pd.Categorical):
            flags[(field, FlagFields.FLAG)] = flags[(field, FlagFields.FLAG)].astype(self.flags)
        return flags
