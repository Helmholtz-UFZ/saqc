#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.categoricalflagger import CategoricalBaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.spike_detection import (
    flagSpikes_spektrumBased,
    flagSpikes_simpleMad,
    flagSpikes_slidingZscore,
    flagSpikes_basic)


TESTFLAGGERS = [
    CategoricalBaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.fixture(scope='module')
def spiky_data():
    index = pd.date_range(start='2011-01-01', end='2011-01-05', freq='5min')
    spiky_series = pd.DataFrame(dict(spiky_data=np.linspace(1, 2, index.size)), index=index)
    spiky_series.iloc[100] = 100
    spiky_series.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return spiky_series, flag_assertion


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSpikes_SpektrumBased(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagSpikes_spektrumBased(data, flags, 'spiky_data', flagger)
    flag_result = flag_result.iloc[:, 0]
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagMad(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagSpikes_simpleMad(data, flags, 'spiky_data', flagger, '1H')
    flag_result = flag_result.iloc[:, 0]
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('method', ['modZ', 'zscore'])
def test_slidingOutlier(spiky_data, flagger, method):

    # test for numeric input
    data = spiky_data[0]
    flags = flagger.initFlags(data)

    tests = [
        flagSpikes_slidingZscore(data, flags, 'spiky_data', flagger, winsz=300, dx=50, method=method),
        flagSpikes_slidingZscore(data, flags, 'spiky_data', flagger, winsz='1500min', dx='250min', method=method)
    ]

    for _, test_flags in tests:
        flag_result = flagger.getFlags(test_flags)
        test_sum = (flag_result.iloc[spiky_data[1]] == flagger.BAD).sum()
        assert int(test_sum) == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSpikes_Basic(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagSpikes_basic(data, flags, 'spiky_data', flagger, thresh=60, tol=10, length='20min')
    flag_result = flag_result.iloc[:, 0]
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])
