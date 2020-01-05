#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.funcs.constants_detection import flagConstantVarianceBased, flagConstant
from saqc.funcs.soil_moisture_tests import flagSoilMoistureConstant

from test.common import TESTFLAGGER, initData


@pytest.fixture
def data():
    constants_data = initData(
        1, start_date="2011-01-01 00:00:00", end_date="2011-01-01 03:00:00", freq="5min"
    )
    constants_data.iloc[5:25] = 0
    return constants_data


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagConstants(data, flagger):
    idx = np.array([5, 6, 7, 8, 9, 10, 18, 19, 20, 21])
    data.iloc[idx] = 200
    expected = np.arange(5, 22)
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result = flagConstant(
        data, field, flagger, window="15Min", thresh=.1,
    )
    flags = flagger_result.getFlags(field)
    assert np.all(flags[expected] == flagger.BAD)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagConstantsVarianceBased(data, flagger):
    data.iloc[5:25] = 200
    expected = np.arange(5, 25)
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    data, flagger_result1 = flagConstantVarianceBased(
        data, field, flagger, window="1h"
    )

    flag_result1 = flagger_result1.getFlags(field)
    test_sum = (flag_result1[expected] == flagger.BAD).sum()
    assert test_sum == len(expected)
