#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Union, Tuple

import pandas as pd
from dios.dios import DictOfSeries
from typing_extensions import Literal

from saqc.core import Flags as Flagger
from saqc.constants import *
from saqc.core.modules.base import ModuleBase
from saqc.lib.types import ColumnName


class FlagTools(ModuleBase):

    def clearFlags(self, field: ColumnName, **kwargs) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("clearFlags", locals())

    def forceFlags(
            self, field: ColumnName, flag: float = BAD, **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("forceFlags", locals())

    def flagDummy(self, field: ColumnName,  **kwargs) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagDummy", locals())

    def flagForceFail(self, field: ColumnName, **kwargs):
        return self.defer("flagForceFail", locals())

    def flagUnflagged(
            self, field: ColumnName, flag: float = BAD, **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagUnflagged", locals())

    def flagGood(self, field: ColumnName, flag=BAD, **kwargs) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagGood", locals())

    def flagManual(
            self, field: ColumnName, 
            mdata: Union[pd.Series, pd.DataFrame, DictOfSeries],
            mflag: Any = 1,
            method: Literal["plain", "ontime", "left-open", "right-open"] = 'plain',
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagManual", locals())
