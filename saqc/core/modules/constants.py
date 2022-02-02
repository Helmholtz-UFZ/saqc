#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from saqc.constants import BAD
import saqc


class Constants:
    def flagByVariance(
        self,
        field: str,
        window: str = "12h",
        thresh: float = 0.0005,
        maxna: int = None,
        maxna_group: int = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByVariance", locals())

    def flagConstants(
        self, field: str, thresh: float, window: str, flag: float = BAD, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagConstants", locals())
