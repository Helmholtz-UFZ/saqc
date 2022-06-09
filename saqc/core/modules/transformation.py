#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Union

import pandas as pd

import saqc
import saqc.funcs
from saqc.lib.docurator import doc


class Transformation:
    @doc(saqc.funcs.transformation.transform.__doc__)
    def transform(
        self,
        field: str,
        func: Callable[[pd.Series], pd.Series],
        freq: Optional[Union[float, str]] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("transform", locals())
