#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union

from dios import DictOfSeries
from typing_extensions import Literal

from saqc.constants import BAD
import saqc
from sphinxdoc.scripts.templates import doc
import saqc.funcs


class Curvefit:

    @doc(saqc.funcs.curvefit.fitPolynomial.__doc__)
    def fitPolynomial(
        self,
        field: str,
        window: Union[int, str],
        order: int,
        min_periods: int = 0,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("fitPolynomial", locals())
