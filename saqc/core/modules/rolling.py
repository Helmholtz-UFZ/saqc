#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from typing import Union, Callable

import numpy as np
import pandas as pd

from saqc.constants import BAD


class Rolling:
    def roll(
        self,
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        **kwargs
    ):
        return self._defer("roll", locals())
