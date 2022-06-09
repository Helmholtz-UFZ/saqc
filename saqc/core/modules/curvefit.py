#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Union

from typing_extensions import Literal

import saqc
import saqc.funcs
from dios import DictOfSeries
from saqc.constants import BAD
from saqc.lib.docurator import doc


class Curvefit:
    @doc(saqc.funcs.curvefit.fitPolynomial.__doc__)
    def fitPolynomial(
        self,
        field: str,
        window: int | str,
        order: int,
        min_periods: int = 0,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("fitPolynomial", locals())
