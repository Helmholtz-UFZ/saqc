#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.constants import *
from saqc.funcs.constants import flagConstants, flagByVariance
from saqc.core import initFlagsLike, Flags

from tests.common import initData


@pytest.fixture
def data():
    constants_data = initData(1, start_date="2011-01-01 00:00:00", end_date="2011-01-01 03:00:00", freq="5min")
    constants_data.iloc[5:25] = 200
    return constants_data


def test_constants_flagBasic(data):
    expected = np.arange(5, 22)
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result = flagConstants(data, field, flags, window="15Min", thresh=0.1, flag=BAD)
    flagscol = flags_result[field]
    assert np.all(flagscol[expected] == BAD)


def test_constants_flagVarianceBased(data):
    expected = np.arange(5, 25)
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result1 = flagByVariance(data, field, flags, window="1h", flag=BAD)

    flag_result1 = flags_result1[field]
    test_sum = (flag_result1[expected] == BAD).sum()
    assert test_sum == len(expected)
