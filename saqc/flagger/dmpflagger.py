#! /usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import json
import pandas as pd

from .simpleflagger import SimpleFlagger
from .baseflagger import BaseFlagger, PandasLike
from ..lib.tools import *


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
        self.flags_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]
        version = subprocess.run("git describe --tags --always --dirty",
                                 shell=True, check=False, stdout=subprocess.PIPE).stdout
        self.project_version = version.decode().strip()

    def initFlags(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        check_isdf(data, 'data', allow_multiindex=False)
        colindex = pd.MultiIndex.from_product(
            [data.columns, self.flags_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])
        flags = pd.DataFrame(data=self.categories[0], columns=colindex, index=data.index)
        return self._assureDtype(flags)

    def getFlags(self, flags, field=None, loc=None, iloc=None, **kwargs):
        self._checkFlags(flags, multi=True)
        if field is None:
            flags = flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
        else:
            flags = flags[[(field, FlagFields.FLAG)]]
            flags.columns = [field]
        flags = super().getFlags(flags, field, loc, iloc, **kwargs)
        return flags
        # getFlags(flags, field, loc, iloc, **kwargs)

    def setFlags(self, flags, field, loc=None, iloc=None, flag=None, force=False, comment='', **kwargs):
        comment = json.dumps(dict(comment=comment, commit=self.project_version, test=kwargs.get("func_name", "")))
        return super().setFlags(flags, field, loc, iloc, flag, force, comment=comment, cause='')

    def _setFlags(self, flags, rowindex, field, flag, comment=None, cause=None, **kwargs):
        if comment is None or cause is None:
            raise AssertionError('wrong implemented :/')
        flags.loc[rowindex, field] = flag, cause, comment
        return self._assureDtype(flags, field)

    def clearFlags(self, flags, field, loc=None, iloc=None, **kwargs):
        if field is None:
            raise ValueError('field cannot be None')
        f = self.getFlags(flags, field, loc, iloc, **kwargs)
        return self._setFlags(flags, f.index, field, flag=self.UNFLAGGED, cause='', comment='', **kwargs)

    def _checkFlags(self, flags, multi=False, **kwargs):
        if multi:
            check_isdfmi(flags, argname='flags')
            return flags
        else:
            return super()._checkFlags(flags, **kwargs)

    def _assureDtype(self, flags, field=None):
        if isinstance(flags, pd.Series):
            return flags if self._isFlagsDtype(flags) else flags.astype(self.categories)

        else:  # got a df, recurse
            if field is None:
                for c in flags:
                    flags[c] = self._assureDtype(flags[c])
            else:
                flags[field] = self._assureDtype(flags[field])
        return flags

    def _assureDtype(self, flags, field=None):
        if isinstance(flags, pd.DataFrame) and isinstance(flags.columns, pd.MultiIndex):
            if field is None:
                cols = [c for c in flags.columns if FlagFields.FLAG in c]
            else:
                cols = [(field, FlagFields.FLAG)]
            for c in cols:
                flags[c] = super()._assureDtype(flags[c])
        else:
            flags = super()._assureDtype(flags, field)
        return flags
