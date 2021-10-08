#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Union

import pandas as pd
from dios import DictOfSeries
from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc
from saqc.lib.types import ColumnName


class FlagTools(ModuleBase):
    def clearFlags(self, field: ColumnName, **kwargs) -> saqc.SaQC:
        return self.defer("clearFlags", locals())

    def forceFlags(self, field: ColumnName, flag: float = BAD, **kwargs) -> saqc.SaQC:
        return self.defer("forceFlags", locals())

    def flagDummy(self, field: ColumnName, **kwargs) -> saqc.SaQC:
        return self.defer("flagDummy", locals())

    def flagUnflagged(
        self, field: ColumnName, flag: float = BAD, **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagUnflagged", locals())

    def flagManual(
        self,
        field: ColumnName,
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries],
        mflag: Any = 1,
        method: Literal["plain", "ontime", "left-open", "right-open"] = "plain",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flagManual", locals())
