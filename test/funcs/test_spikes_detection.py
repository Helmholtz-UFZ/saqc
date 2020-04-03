#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.funcs.spikes_detection import (
    spikes_flagSpektrumBased,
    spikes_flagMad,
    spikes_flagSlidingZscore,
    spikes_flagBasic,
    spikes_flagRaise,
    spikes_flagOddWater,
)

from test.common import TESTFLAGGER


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    spiky_series = pd.DataFrame(dict(spiky_data=np.linspace(1, 2, index.size)), index=index)
    spiky_series.iloc[100] = 100
    spiky_series.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return spiky_series, flag_assertion


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSpikesSpektrumBased(spiky_data, flagger):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = spikes_flagSpektrumBased(data, field, flagger)
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagMad(spiky_data, flagger):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = spikes_flagMad(data, field, flagger, "1H")
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("method", ["modZ", "zscore"])
def test_slidingOutlier(spiky_data, flagger, method):

    # test for numeric input
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)

    tests = [
        spikes_flagSlidingZscore(data, field, flagger, window=300, offset=50, method=method),
        spikes_flagSlidingZscore(data, field, flagger, window="1500min", offset="250min", method=method),
    ]

    for _, flagger_result in tests:
        flag_result = flagger_result.getFlags(field)
        test_sum = (flag_result.iloc[spiky_data[1]] == flagger.BAD).sum()
        assert int(test_sum) == len(spiky_data[1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSpikesBasic(spiky_data, flagger):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = spikes_flagBasic(data, field, flagger, thresh=60, tolerance=10, window_size="20min")
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize(
    "dat",
    [
        pytest.lazy_fixture("course_1"),
        pytest.lazy_fixture("course_2"),
        pytest.lazy_fixture("course_3"),
        pytest.lazy_fixture("course_4"),
    ],
)
def test_flagSpikesLimitRaise(dat, flagger):
    data, characteristics = dat()
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    _, flagger_result = spikes_flagRaise(
        data, field, flagger, thresh=2, intended_freq="10min", raise_window="20min", numba_boost=False
    )
    assert flagger_result.isFlagged(field)[characteristics["raise"]].all()
    assert not flagger_result.isFlagged(field)[characteristics["return"]].any()
    assert not flagger_result.isFlagged(field)[characteristics["drop"]].any()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_flagSpikesOddWater(dat, flagger):
    data1, characteristics = dat(periods=1000, initial_level=5, final_level=15, out_val=50)
    data2, characteristics = dat(periods=1000, initial_level=20, final_level=1, out_val=30)
    field = "dummy"
    fields = ["data1", "data2"]
    data = pd.DataFrame({"data1": data1.squeeze(), "data2": data2.squeeze()}, index=data1.index)
    flagger = flagger.initFlags(data)
    _, flagger_result = spikes_flagOddWater(
        data, field, flagger, fields=fields, bin_frac=50, trafo="np.log", iter_start=0.95, n_neighbors=10
    )
    assert flagger_result.isFlagged(fields[0])[characteristics["raise"]].all()
