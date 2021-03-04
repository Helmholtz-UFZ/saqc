#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import pandas as pd

from dios import dios

from saqc.common import *
from saqc.flagger import Flagger, initFlagsLike
from saqc.funcs.pattern import *
from test.common import initData


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.skip(reason='faulty implementation - will get fixed by GL-MR191')
def test_flagPattern_wavelet():
    data = pd.Series(0, index=pd.date_range(start="2000", end='2001', freq='1d'))
    data.iloc[2:4] = 7
    pattern = data.iloc[1:6]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))
    flagger = initFlagsLike(data, name='data')
    data, flagger = flagPatternByDTW(data, "data", flagger, ref_field="pattern_data", flag=BAD)

    assert all(flagger["data"][1:6])
    assert any(flagger["data"][:1])
    assert any(flagger["data"][7:])


@pytest.mark.skip(reason='faulty implementation - will get fixed by GL-MR191')
def test_flagPattern_dtw():
    data = pd.Series(0, index=pd.date_range(start="2000", end='2001', freq='1d'))
    data.iloc[2:4] = 7
    pattern = data.iloc[1:6]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))
    flagger = initFlagsLike(data, name='data')
    data, flagger = flagPatternByWavelet(data, "data", flagger, ref_field="pattern_data", flag=BAD)

    assert all(flagger["data"][1:6])
    assert any(flagger["data"][:1])
    assert any(flagger["data"][7:])
