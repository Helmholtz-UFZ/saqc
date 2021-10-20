#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Union

import pandas as pd
from dios import DictOfSeries
from typing_extensions import Literal

from saqc.constants import BAD
import saqc


class FlagTools:
    def clearFlags(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("clearFlags", locals())

    def forceFlags(self, field: str, flag: float = BAD, **kwargs) -> saqc.SaQC:
        return self._defer("forceFlags", locals())

    def flagDummy(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("flagDummy", locals())

    def flagUnflagged(self, field: str, flag: float = BAD, **kwargs) -> saqc.SaQC:
        return self._defer("flagUnflagged", locals())

    def flagManual(
        self,
        field: str,
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries],
        mflag: Any = 1,
        method: Literal["plain", "ontime", "left-open", "right-open"] = "plain",
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagManual", locals())
