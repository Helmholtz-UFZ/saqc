#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc
import saqc.funcs
from dios import DictOfSeries
from saqc.constants import BAD, FILTER_ALL
from saqc.lib.docurator import doc


class FlagTools:
    @doc(saqc.funcs.flagtools.clearFlags.__doc__)
    def clearFlags(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("clearFlags", locals())

    @doc(saqc.funcs.flagtools.forceFlags.__doc__)
    def forceFlags(self, field: str, flag: float = BAD, **kwargs) -> saqc.SaQC:
        return self._defer("forceFlags", locals())

    @doc(saqc.funcs.flagtools.forceFlags.__doc__)
    def flagDummy(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("flagDummy", locals())

    @doc(saqc.funcs.flagtools.flagUnflagged.__doc__)
    def flagUnflagged(self, field: str, flag: float = BAD, **kwargs) -> saqc.SaQC:
        return self._defer("flagUnflagged", locals())

    @doc(saqc.funcs.flagtools.flagManual.__doc__)
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

    @doc(saqc.funcs.flagtools.transferFlags.__doc__)
    def transferFlags(
        self,
        field: str | Sequence[str],
        target: str | Sequence[str],
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("transferFlags", locals())

    @doc(saqc.funcs.flagtools.propagateFlags.__doc__)
    def propagateFlags(
        self,
        field: str | Sequence[str],
        window: Union[str, int],
        method: Literal["ffill", "bfill"] = "ffill",
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("propagateFlags", locals())
