#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Union

import pandas as pd
import numpy as np
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
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries, list, np.array],
        method: Literal[
            "left-open", "right-open", "closed", "plain", "ontime"
        ] = "left-open",
        mformat: Literal["start-end", "mflag"] = "start-end",
        mflag: Any = 1,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagManual", locals())

    def transferFlags(
        self,
        field: str | Sequence[str],
        target: str | Sequence[str],
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("transferFlags", locals())
