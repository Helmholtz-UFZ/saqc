#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from saqc import SaQC


@pytest.fixture
def data():
    dat = pd.DataFrame(
        {"d" + str(k): np.random.random(1000) for k in range(2)},
        index=pd.date_range("2000", freq="10min", periods=1000),
    )
    dat.iloc[np.random.randint(0, 1000, 10), 0] = np.nan
    return dat


@pytest.mark.parametrize("field", ["d0", ["d1", "d0"]])
@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("context", [512, 256])
def test_fitFMmoment(data, field, ratio, context):
    qc = SaQC(data)
    qc.fitMomentFM(field, ratio, context)
