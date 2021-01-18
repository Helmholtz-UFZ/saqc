#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from dios import dios

from saqc.funcs.pattern import *
from test.common import initData, TESTFLAGGER


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagPattern_wavelet(flagger):

    data = pd.Series(0, index=pd.date_range(start="2000", end='2001', freq='1d'))
    data.iloc[2:4] = 7
    pattern = data.iloc[1:6]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))

    flagger = flagger.initFlags(data)
    data, flagger = flagPatternByDTW(data, "data", flagger, ref_field="pattern_data")
    assert (flagger.isFlagged("data")[1:6]).all()
    assert (flagger.isFlagged("data")[:1]).any()
    assert (flagger.isFlagged("data")[7:]).any()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagPattern_dtw(flagger):

    data = pd.Series(0, index=pd.date_range(start="2000", end='2001', freq='1d'))
    data.iloc[2:4] = 7
    pattern = data.iloc[1:6]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))

    flagger = flagger.initFlags(data)
    data, flagger = flagPatternByWavelet(data, "data", flagger, ref_field="pattern_data")
    assert (flagger.isFlagged("data")[1:6]).all()
    assert (flagger.isFlagged("data")[:1]).any()
    assert (flagger.isFlagged("data")[7:]).any()
