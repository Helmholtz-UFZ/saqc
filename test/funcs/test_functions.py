#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.funcs.functions import (
    flagRange,
    flagSesonalRange,
    forceFlags,
    clearFlags,
    flagIsolated,
)
from test.common import initData, TESTFLAGGER


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagRange(data, field, flagger):
    min, max = 10, 90
    flagger = flagger.initFlags(data)
    data, flagger = flagRange(data, field, flagger, min=min, max=max)
    flagged = flagger.isFlagged(field)
    expected = (data[field] < min) | (data[field] > max)
    assert np.all(flagged == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSesonalRange(data, field, flagger):
    # prepare
    data.loc[::2] = 0
    data.loc[1::2] = 50
    nyears = len(data.index.year.unique())

    tests = [
        ({"min": 1, "max": 100, "startmonth": 7, "startday": 1, "endmonth": 8, "endday": 31,}, 31 * 2 * nyears // 2,),
        ({"min": 1, "max": 100, "startmonth": 12, "startday": 16, "endmonth": 1, "endday": 15,}, 31 * nyears // 2 + 1,),
    ]

    for test, expected in tests:
        flagger = flagger.initFlags(data)
        data, flagger = flagSesonalRange(data, field, flagger, **test)
        flagged = flagger.isFlagged(field)
        assert flagged.sum() == expected


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_clearFlags(data, field, flagger):
    flagger = flagger.initFlags(data)
    flags_orig = flagger.getFlags()
    flags_set = flagger.setFlags(field, flag=flagger.BAD).getFlags()
    _, flagger = clearFlags(data, field, flagger)
    flags_cleared = flagger.getFlags()
    assert np.all(flags_orig != flags_set)
    assert np.all(flags_orig == flags_cleared)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_forceFlags(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns
    flags_orig = flagger.setFlags(field).getFlags(field)
    _, flagger = forceFlags(data, field, flagger, flag=flagger.GOOD)
    flags_forced = flagger.getFlags(field)
    assert np.all(flags_orig != flags_forced)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagIsolated(data, flagger):
    field = data.columns[0]
    data.iloc[1:3, 0] = np.nan
    data.iloc[4:5, 0] = np.nan
    data.iloc[11:13, 0] = np.nan
    data.iloc[15:17, 0] = np.nan
    flagger = flagger.initFlags(data)
    flagger = flagger.setFlags(field, iloc=slice(5, 6))

    _, flagger_result = flagIsolated(data, field, flagger, group_window="1D", gap_window="2.1D")

    assert flagger_result.isFlagged(field)[slice(3, 6, 2)].all()

    flagger = flagger.setFlags(field, iloc=slice(3, 4), flag=flagger.UNFLAGGED, force=True)
    data, flagger_result = flagIsolated(
        data, field, flagger_result, group_window="2D", gap_window="2.1D", continuation_range="1.1D",
    )
    assert flagger_result.isFlagged(field)[[3, 5, 13, 14]].all()
