#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from saqc import BAD, UNFLAGGED, SaQC
from saqc.core import DictOfSeries, initFlagsLike
from tests.common import initData


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_flagPattern_dtw(plot, normalize):
    data = pd.Series(0, index=pd.date_range(start="2000", end="2001", freq="1d"))
    data.iloc[10:18] = [0, 5, 6, 7, 6, 8, 5, 0]
    pattern = data.iloc[10:18]

    data = DictOfSeries(data=data, pattern_data=pattern)
    flags = initFlagsLike(data, name="data")
    qc = SaQC(data, flags).flagPatternByDTW(
        "data",
        reference="pattern_data",
        plot=plot,
        normalize=normalize,
        flag=BAD,
    )

    assert all(qc.flags["data"].iloc[10:18] == BAD)
    assert all(qc.flags["data"].iloc[:9] == UNFLAGGED)
    assert all(qc.flags["data"].iloc[18:] == UNFLAGGED)
