#! /usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import json
from copy import deepcopy
from collections import OrderedDict
from typing import Union, Sequence

import pandas as pd

from saqc.flagger.categoricalflagger import CategoricalFlagger
from saqc.lib.tools import assertDataFrame, toSequence, assertScalar


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


class DmpFlagger(CategoricalFlagger):
    def __init__(self):
        super().__init__(FLAGS)
        self.flags_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]
        version = subprocess.run(
            "git describe --tags --always --dirty", shell=True, check=False, stdout=subprocess.PIPE,
        ).stdout
        self.project_version = version.decode().strip()
        self.signature = ("flag", "comment", "cause", "force")
        self._flags = None

    def initFlags(self, data: pd.DataFrame = None, flags: pd.DataFrame = None):
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        if data is not None:
            flags = pd.DataFrame(data="", columns=self._getColumnIndex(data.columns), index=data.index,)
            flags.loc[:, self._getColumnIndex(data.columns, [FlagFields.FLAG])] = self.UNFLAGGED
        elif flags is not None:
            if not isinstance(flags.columns, pd.MultiIndex):
                cols = flags.columns
                flags = flags.copy()
                flags.columns = self._getColumnIndex(cols, [FlagFields.FLAG])
                flags = flags.reindex(columns=self._getColumnIndex(cols), fill_value="")
        else:
            raise TypeError("either 'data' or 'flags' are required")

        return self._copy(self._assureDtype(flags))

    def getFlagger(self, field=None, loc=None, iloc=None):
        # NOTE: we need to preserve all indexing levels
        assertScalar("field", field, optional=True)
        variables = self._flags.columns.get_level_values(ColumnLevels.VARIABLES).drop_duplicates()
        cols = toSequence(field, variables)
        out = super().getFlagger(field, loc, iloc)
        out._flags.columns = self._getColumnIndex(cols)
        return out

    def getFlags(self, field=None, loc=None, iloc=None):
        assertScalar("field", field, optional=True)
        field = field or slice(None)
        mask = self._locatorMask(field, loc, iloc)
        flags = self._flags.xs(FlagFields.FLAG, level=ColumnLevels.FLAGS, axis=1).copy()
        return super()._assureDtype(flags.loc[mask, field])

    def setFlags(self, field, loc=None, iloc=None, flag=None, force=False, comment="", cause="", **kwargs):
        assertScalar("field", field, optional=True)

        flag = self.BAD if flag is None else self._checkFlag(flag)

        comment = json.dumps({"comment": comment, "commit": self.project_version, "test": kwargs.get("func_name", ""),})

        this = self.getFlags(field=field)
        other = self._broadcastFlags(field=field, flag=flag)
        mask = self._locatorMask(field, loc, iloc)
        if not force:
            mask &= (this < other).values

        out = deepcopy(self)
        out._flags.loc[mask, field] = other[mask], cause, comment
        return out

    def _getColumnIndex(
        self, cols: Union[str, Sequence[str]], fields: Union[str, Sequence[str]] = None
    ) -> pd.MultiIndex:
        cols = toSequence(cols)
        fields = toSequence(fields, self.flags_fields)
        return pd.MultiIndex.from_product([cols, fields], names=[ColumnLevels.VARIABLES, ColumnLevels.FLAGS])

    def _assureDtype(self, flags):
        # NOTE: building up new DataFrames is significantly
        #       faster than assigning into existing ones
        tmp = OrderedDict()
        for (var, flag_field) in flags.columns:
            col_data = flags[(var, flag_field)]
            if flag_field == FlagFields.FLAG:
                col_data = col_data.astype(self.dtype)
            else:
                col_data = col_data.astype(str)
            tmp[(var, flag_field)] = col_data
        return pd.DataFrame(tmp, columns=flags.columns, index=flags.index)
