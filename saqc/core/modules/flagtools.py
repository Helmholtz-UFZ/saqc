#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union
from typing_extensions import Literal

import pandas as pd

from dios.dios import DictOfSeries
from saqc.core.modules.base import ModuleBase
from saqc.common import *


class FlagTools(ModuleBase):

    def clearFlags(self, field: str, **kwargs):
        return self.defer("clearFlags", locals())

    def forceFlags(self, field: str, flag: float = BAD, **kwargs):
        return self.defer("forceFlags", locals())

    def flagDummy(self, field: str, **kwargs):
        return self.defer("flagDummy", locals())

    def flagForceFail(self, field: str, **kwargs):
        return self.defer("flagForceFail", locals())

    def flagUnflagged(self, field: str, flag: float = BAD, **kwargs):
        return self.defer("flagUnflagged", locals())

    def flagGood(self, field: str, flag: float = BAD, **kwargs):
        return self.defer("flagGood", locals())

    def flagManual(
            self,
            field: str,
            mdata: Union[pd.Series, pd.DataFrame, DictOfSeries],
            mflag: Any = 1,
            method=Literal["plain", "ontime", "left-open", "right-open"],
            **kwargs
    ):
        return self.defer("flagManual", locals())
