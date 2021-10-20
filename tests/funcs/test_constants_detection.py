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
    constants_data = initData(
        1, start_date="2011-01-01 00:00:00", end_date="2011-01-01 03:00:00", freq="5min"
    )
    constants_data.iloc[5:25] = 200
    return constants_data


def test_constants_flagBasic(data):
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result = flagConstants(
        data, field, flags, window="15Min", thresh=0.1, flag=BAD
    )
    flagscol = flags_result[field]
    assert np.all(flagscol[5:25] == BAD)
    assert np.all(flagscol[:5] == UNFLAGGED)
    assert np.all(flagscol[25 + 1 :] == UNFLAGGED)


def test_constants_flagVarianceBased(data):
    expected = np.arange(5, 25)
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result1 = flagByVariance(data, field, flags, window="1h", flag=BAD)

    flagscol = flags_result1[field]
    assert np.all(flagscol[5:25] == BAD)
    assert np.all(flagscol[:5] == UNFLAGGED)
    assert np.all(flagscol[25 + 1 :] == UNFLAGGED)
