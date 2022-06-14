#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import saqc
import saqc.funcs
from saqc.constants import BAD, FILTER_ALL
from saqc.lib.docurator import doc


class Breaks:
    @doc(saqc.funcs.breaks.flagMissing.__doc__)
    def flagMissing(
        self, field: str, flag: float = BAD, dfilter: float = FILTER_ALL, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagMissing", locals())

    @doc(saqc.funcs.breaks.flagIsolated.__doc__)
    def flagIsolated(
        self,
        field: str,
        gap_window: str,
        group_window: str,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagIsolated", locals())

    @doc(saqc.funcs.breaks.flagJumps.__doc__)
    def flagJumps(
        self,
        field: str,
        thresh: float,
        window: str,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagJumps", locals())
