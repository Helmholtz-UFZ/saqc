#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import saqc
import saqc.funcs
from saqc.constants import BAD
from saqc.lib.docurator import doc


class Constants:
    @doc(saqc.funcs.constants.flagByVariance.__doc__)
    def flagByVariance(
        self,
        field: str,
        window: str,
        thresh: float,
        maxna: int | None = None,
        maxna_group: int | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagByVariance", locals())

    @doc(saqc.funcs.constants.flagConstants.__doc__)
    def flagConstants(
        self, field: str, thresh: float, window: int | str, flag: float = BAD, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagConstants", locals())
