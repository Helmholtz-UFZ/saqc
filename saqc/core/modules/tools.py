#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import numpy as np
from typing_extensions import Literal

import saqc
import saqc.funcs
from saqc.constants import FILTER_NONE
from saqc.lib.docurator import doc


class Tools:
    @doc(saqc.funcs.tools.copyField.__doc__)
    def copyField(
        self, field: str, target: str, overwrite: bool = False, **kwargs
    ) -> saqc.SaQC:
        return self._defer("copyField", locals())

    @doc(saqc.funcs.tools.dropField.__doc__)
    def dropField(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("dropField", locals())

    @doc(saqc.funcs.tools.renameField.__doc__)
    def renameField(self, field: str, new_name: str, **kwargs) -> saqc.SaQC:
        return self._defer("renameField", locals())

    @doc(saqc.funcs.tools.selectTime.__doc__)
    def selectTime(
        self,
        field: str,
        mode: Literal["periodic", "selection_field"],
        selection_field: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        closed: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("selectTime", locals())

    @doc(saqc.funcs.tools.plot.__doc__)
    def plot(
        self,
        field: str,
        path: Optional[str] = None,
        max_gap: Optional[str] = None,
        history: Optional[Literal["valid", "complete"] | list] = "valid",
        xscope: Optional[slice] = None,
        phaseplot: Optional[str] = None,
        store_kwargs: Optional[dict] = None,
        ax_kwargs: Optional[dict] = None,
        dfilter: Optional[float] = FILTER_NONE,
        **kwargs,
    ) -> saqc.SaQC:

        return self._defer("plot", locals())
