#! /usr/bin/env python
# -*- coding: utf-8 -*-

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest
import numpy as np
import pandas as pd
import dios
from test.fixtures import *

from saqc.funcs.outliers import (
    flagMAD,
    flagOffset,
    flagRaise,
    flagMVScores,
    flagByGrubbs,
)
from saqc.common import *
from saqc.flagger import Flagger, initFlagsLike


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    s = pd.Series(np.linspace(1, 2, index.size), index=index, name="spiky_data")
    s.iloc[100] = 100
    s.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return dios.DictOfSeries(s), flag_assertion


def test_flagMad(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = initFlagsLike(data)
    data, flagger_result = flagMAD(data, field, flagger, "1H", flag=BAD)
    flag_result = flagger_result[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


def test_flagSpikesBasic(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = initFlagsLike(data)
    data, flagger_result = flagOffset(data, field, flagger, thresh=60, tolerance=10, window="20min", flag=BAD)
    flag_result = flagger_result[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


# see test/functs/fixtures.py for the 'course_N'
@pytest.mark.parametrize(
    "dat",
    [
        pytest.lazy_fixture("course_1"),
        pytest.lazy_fixture("course_2"),
        pytest.lazy_fixture("course_3"),
        pytest.lazy_fixture("course_4"),
    ],
)
def test_flagSpikesLimitRaise(dat):
    data, characteristics = dat()
    field, *_ = data.columns
    flagger = initFlagsLike(data)
    _, flagger_result = flagRaise(
        data, field, flagger,
        thresh=2, intended_freq="10min", raise_window="20min", numba_boost=False, flag=BAD
    )
    assert np.all(flagger_result[field][characteristics["raise"]] > UNFLAGGED)
    assert not np.any(flagger_result[field][characteristics["return"]] > UNFLAGGED)
    assert not np.any(flagger_result[field][characteristics["drop"]] > UNFLAGGED)


# see test/functs/fixtures.py for the 'course_N'
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_flagMultivarScores(dat):
    data1, characteristics = dat(periods=1000, initial_level=5, final_level=15, out_val=50)
    data2, characteristics = dat(periods=1000, initial_level=20, final_level=1, out_val=30)
    field = "dummy"
    fields = ["data1", "data2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["data1", "data2"])
    flagger = initFlagsLike(data)
    _, flagger_result = flagMVScores(
        data, field, flagger, fields=fields, trafo=np.log, iter_start=0.95, n_neighbors=10, flag=BAD
    )
    for field in fields:
        isflagged = flagger_result[field] > UNFLAGGED
        assert isflagged[characteristics["raise"]].all()
        assert not isflagged[characteristics["return"]].any()
        assert not isflagged[characteristics["drop"]].any()


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_grubbs(dat):
    data, char_dict = dat(
        freq="10min", periods=45,
        initial_level=0, final_level=0,
        crowd_size=1, crowd_spacing=3,
        out_val=-10,
    )
    flagger = initFlagsLike(data)
    data, result_flagger = flagByGrubbs(data, "data", flagger, winsz=20, min_periods=15, flag=BAD)
    assert np.all(result_flagger["data"][char_dict["drop"]] > UNFLAGGED)

