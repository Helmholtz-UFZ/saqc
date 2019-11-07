#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.spike_detection import flagSpikes_SpektrumBased, flagMad, slidingOutlier, flagSpikes_Basic

from saqc.lib.tools import getPandasData

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.fixture(scope='module')
def spiky_data():
    index = pd.date_range(start='2011-01-01', end='2011-01-05', freq='5min')
    spiky_series = pd.Series(np.linspace(1, 2, index.size), index=index, name='spiky_data')
    spiky_series.iloc[100] = 100
    spiky_series.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return spiky_series, flag_assertion


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSpikes_SpektrumBased(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data.to_frame())
    data, flag_result = flagSpikes_SpektrumBased(data, flags, 'spiky_data', flagger)
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagMad(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data.to_frame())
    data, flag_result = flagMad(data, flags, 'spiky_data', flagger, '1H')
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('method', ['modZ', 'zscore'])
def test_slidingOutlier(spiky_data, flagger, method):

    # test for numeric input
    data = spiky_data[0].to_frame()
    flags = flagger.initFlags(data)
    _, flag_result = slidingOutlier(data, flags, 'spiky_data', flagger, winsz=300, dx=50, method=method)
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])

    # test for offset input
    _, flag_result = slidingOutlier(data, flags, 'spiky_data', flagger, winsz='1500min', dx='250min', method=method)
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSpikes_Basic(spiky_data, flagger):
    data = spiky_data[0]
    flags = flagger.initFlags(data.to_frame())
    data, flag_result = flagSpikes_Basic(data, flags, 'spiky_data', flagger, thresh=60, tol=10, length='20min')
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[spiky_data[1]] == flagger.BAD).sum()
    assert test_sum == len(spiky_data[1])
