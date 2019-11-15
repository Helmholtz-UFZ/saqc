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
        self.signature = ("flag", "comment", "cause", "force")
        self._flags = None

    def initFlags(self, data: pd.DataFrame, **kwargs):
        check_isdf(data, 'data', allow_multiindex=False)
        colindex = pd.MultiIndex.from_product(
            [data.columns, self.flags_fields],
            names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])
        flags = pd.DataFrame(data=self.UNFLAGGED, columns=colindex, index=data.index)
        self._flags = flags

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, comment='', cause='', **kwargs):
        comment = json.dumps(dict(comment=comment, commit=self.project_version, test=kwargs.get("func_name", "")))
        # call is redirected to self._writeFlags()
        return super().setFlags(field, loc, iloc, flag, force, comment=comment, cause=cause)

    def clearFlags(self, field, loc=None, iloc=None, **kwargs):
        # call is redirected to self._writeFlags()
        kwargs.pop('cause', None), kwargs.pop('comment', None)

        super().clearFlags(field, loc=loc, iloc=iloc, cause=self.UNFLAGGED, comment=self.UNFLAGGED, **kwargs)

        ret = self._reduceColumns(self._flags, field, **kwargs)
        return self._assureDtype(ret, field, **kwargs)

    def _writeFlags(self, rowindex, field, flag, cause=None, comment=None, **kwargs):
        assert comment is not None and cause is not None
        self._flags.loc[rowindex, field] = flag, cause, comment

    def _reduceColumns(self, dfmi, field=None, loc=None, iloc=None, **kwargs):
        df = dfmi.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1)
        return df

    def _assureDtype(self, flags, field=None, **kwargs):
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
