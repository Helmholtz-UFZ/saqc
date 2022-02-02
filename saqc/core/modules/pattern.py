#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from saqc.constants import BAD
import saqc


class Pattern:
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
