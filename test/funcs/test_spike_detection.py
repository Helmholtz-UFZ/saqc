#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.funcs.spike_detection import (
    flagSpikes_spektrumBased,
    flagSpikes_simpleMad,
    flagSpikes_slidingZscore,
    flagSpikes_basic,
)

from test.common import TESTFLAGGER


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    spiky_series = pd.DataFrame(
        dict(spiky_data=np.linspace(1, 2, index.size)), index=index
    )
    spiky_series.iloc[100] = 100
    spiky_series.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return spiky_series, flag_assertion


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSpikesSpektrumBased(spiky_data, flagger):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = flagSpikes_spektrumBased(data, field, flagger)
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagMad(spiky_data, flagger):
    data = spiky_data[0]
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = flagSpikes_simpleMad(data, field, flagger, "1H")
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
        flagSpikes_slidingZscore(data, field, flagger, window=300, offset=50, method=method),
        flagSpikes_slidingZscore(
            data, field, flagger, window="1500min", offset="250min", method=method
        ),
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
    data, flagger_result = flagSpikes_basic(
        data, field, flagger, thresh=60, tolerance=10, window_size="20min"
    )
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])
