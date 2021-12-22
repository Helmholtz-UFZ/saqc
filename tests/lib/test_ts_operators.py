# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from saqc.lib.ts_operators import butterFilter
import pandas as pd


def test_butterFilter():
    assert (
        butterFilter(pd.Series([1, -1] * 100), cutoff=0.1) - pd.Series([1, -1] * 100)
    ).mean() < 0.5
