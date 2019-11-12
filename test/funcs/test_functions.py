#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags

from test.common import initData, TESTFLAGGER


@pytest.fixture
def data():
    return initData(
        cols=1,
        start_date="2016-01-01", end_date="2018-12-31",
        freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_range(data, field, flagger):
    min, max = 10, 90
    flags = flagger.initFlags(data)
    data, flags = flagRange(data, flags, field, flagger, min=min, max=max)
    flagged = flagger.isFlagged(flags[field])
    expected = (data[field] < min) | (data[field] >= max)
    assert np.all(flagged == expected)


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_flagSesonalRange(data, field, flagger):
    # prepare
    data.loc[::2] = 0
    data.loc[1::2] = 50
    nyears = len(data.index.year.unique())

    tests = [
        ({"min": 1, "max": 100, "startmonth": 7, "startday": 1, "endmonth": 8, "endday": 31},
         31*2*nyears//2),
        ({"min": 1, "max": 100, "startmonth": 12, "startday": 16, "endmonth": 1, "endday": 15},
         31*nyears//2 + 1)
    ]

    for test, expected in tests:
        flags = flagger.initFlags(data)
        data, flags = flagSesonalRange(data, flags, field, flagger, **test)
        flagged = flagger.isFlagged(flags[field])
        assert flagged.sum() == expected


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_clearFlags(data, field, flagger):
    orig = flagger.initFlags(data)
    flags = flagger.setFlags(orig.copy(), field, flag=flagger.BAD)
    _, cleared = clearFlags(data, flags, field, flagger)
    assert np.all(orig != flags)
    assert np.all(orig == cleared)


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_forceFlags(data, flagger):
    field, *_ = data.columns
    flags = flagger.setFlags(flagger.initFlags(data), field)
    orig = flags.copy()
    _, forced = forceFlags(data, flags, field, flagger, flag=flagger.GOOD)
    assert np.all(flagger.getFlags(orig) != flagger.getFlags(forced))
