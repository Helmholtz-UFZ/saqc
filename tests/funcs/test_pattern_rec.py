#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import dios

from saqc.constants import *
from saqc.core import initFlagsLike
from saqc.funcs.pattern import *
from tests.common import initData


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.skip(reason="faulty implementation - wait for #GL216")
def test_flagPattern_wavelet():
    data = pd.Series(0, index=pd.date_range(start="2000", end="2001", freq="1d"))
    data.iloc[10:18] = [0, 5, 6, 7, 6, 8, 5, 0]
    pattern = data.iloc[10:18]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))
    flags = initFlagsLike(data, name="data")
    data, flags = flagPatternByWavelet(
        data, "data", flags, ref_field="pattern_data", flag=BAD
    )

    assert all(flags["data"].iloc[10:18] == BAD)
    assert all(flags["data"].iloc[:9] == UNFLAGGED)
    assert all(flags["data"].iloc[18:] == UNFLAGGED)


def test_flagPattern_dtw():
    data = pd.Series(0, index=pd.date_range(start="2000", end="2001", freq="1d"))
    data.iloc[10:18] = [0, 5, 6, 7, 6, 8, 5, 0]
    pattern = data.iloc[10:18]

    data = dios.DictOfSeries(dict(data=data, pattern_data=pattern))
    flags = initFlagsLike(data, name="data")
    data, flags = flagPatternByDTW(
        data, "data", flags, ref_field="pattern_data", flag=BAD
    )

    assert all(flags["data"].iloc[10:18] == BAD)
    assert all(flags["data"].iloc[:9] == UNFLAGGED)
    assert all(flags["data"].iloc[18:] == UNFLAGGED)

    # visualize:
    # data['data'].plot()
    # ((flags['data']>0) *5.).plot()
