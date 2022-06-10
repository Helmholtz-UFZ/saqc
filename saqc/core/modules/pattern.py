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


class Pattern:
    @doc(saqc.funcs.pattern.flagPatternByDTW.__doc__)
    def flagPatternByDTW(
        self,
        field,
        reference,
        max_distance=0.0,
        normalize=True,
        plot=False,
        flag=BAD,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagPatternByDTW", locals())
