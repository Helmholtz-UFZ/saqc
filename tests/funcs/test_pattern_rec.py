#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import pytest

from saqc import BAD, UNFLAGGED, DictOfSeries, SaQC
from saqc.core.flags import initFlagsLike
from tests.common import initData


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_flagPlateau():
    path = os.path.join(
        os.path.abspath(""), "docs/resources/data/turbidity_plateaus.csv"
    )
    dat = pd.read_csv(path, parse_dates=[0], index_col=0)
    dat = dat.interpolate("linear")
    dat = dat.ffill().bfill()
    qc = SaQC(dat)
    qc = qc.flagPlateau(
        "base3", min_length="10min", max_length="7d", granularity="20min"
    )
    anomalies = [
        (0, 0),
        (5313, 5540),
        (10000, 10200),
        (15000, 15500),
        (17000, 17114),
        (17790, 17810),
    ]
    f = qc["base3"].flags.to_pandas().squeeze() > 0
    for i in range(1, len(anomalies)):
        a_slice = slice(anomalies[i][0], anomalies[i][1])
        na_slice = slice(anomalies[i - 1][1], anomalies[i][0])
        assert f.iloc[a_slice].all()
        assert not (f.iloc[na_slice].any())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_flagPlateau_long():
    path = os.path.join(
        os.path.abspath(""), "docs/resources/data/turbidity_plateaus.csv"
    )
    dat = pd.read_csv(path, parse_dates=[0], index_col=0)
    dat = dat.interpolate("linear")
    dat = dat.ffill().bfill()
    _long = np.append(dat.values, [dat.values] * 10)
    dat = pd.Series(
        _long,
        index=pd.date_range("2000", freq="10min", periods=len(_long)),
        name="base3",
    )
    qc = SaQC(dat)
    qc = qc.flagPlateau(
        "base3", min_length="10min", max_length="7d", granularity="20min"
    )
    anomalies = [
        (0, 0),
        (5313, 5540),
        (10000, 10200),
        (15000, 15500),
        (17000, 17114),
        (17790, 17810),
    ]
    f = qc["base3"].flags.to_pandas().squeeze() > 0
    for i in range(1, len(anomalies)):
        a_slice = slice(anomalies[i][0], anomalies[i][1])
        na_slice = slice(anomalies[i - 1][1], anomalies[i][0])
        assert f.iloc[a_slice].all()
        assert not (f.iloc[na_slice].any())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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
