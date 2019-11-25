#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.core.evaluator import evalExpression
from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags, flagIsolated
from saqc.flagger.dmpflagger import DmpFlagger
from test.common import initData, TESTFLAGGER, initMetaDict


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
def test_flagAfter(data, field, flagger):
    flags = flagger.initFlags(data)

    min = data.iloc[int(len(data)*.3), 0]
    max = data.iloc[int(len(data)*.6), 0]
    _, range_flags = flagRange(data, flags, field, flagger, min, max)

    tests = [
        (f"flagWindowAfterFlag(window='3D', func=range(min={min}, max={max}))", "3D"),
        (f"flagNextAfterFlag(n=4, func=range(min={min}, max={max}))", 4),
    ]

    for expr, window in tests:
        _, repeated_flags = evalExpression(expr, data, flags, field, flagger)
        flagged = repeated_flags[flagger.isFlagged(flags)].dropna()
        flag_groups = (flagged
                       .rolling(window=window)
                       .apply(lambda df: flagger.isFlagged(flags).all(), raw=False))
        assert np.all(flag_groups)


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_range(data, field, flagger):
    min, max = 10, 90
    flags = flagger.initFlags(data)
    data, flags = flagRange(data, flags, field, flagger, min=min, max=max)
    flagged = flagger.isFlagged(flags, field)
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
        flagged = flagger.isFlagged(flags, field)
        assert flagged.sum() == expected


@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_clearFlags(data, field, flagger):
    orig = flagger.initFlags(data)
    flags = flagger.setFlags(orig, field, flag=flagger.BAD)
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

@pytest.mark.parametrize('flagger', TESTFLAGGER)
def test_flagIsolated(data, flagger):
    field = data.columns[0]
    data.iloc[1:3, 0] = np.nan
    data.iloc[4:5, 0] = np.nan
    data.iloc[11:13, 0] = np.nan
    data.iloc[15:17, 0] = np.nan
    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, field, iloc=slice(5, 6))
    data, flags = flagIsolated(data, flags, field, flagger, '2.1D', drop_flags='BAD')

    assert flagger.isFlagged(flags,field)[slice(3, 6, 2)].all()

    flags = flagger.setFlags(flags, field, iloc=slice(3,4), flag=flagger.UNFLAGGED, force=True)
    data, flags = flagIsolated(data, flags, field, flagger, '2.1D', max_isolated_group_size=2,
                               continuation_range='1.1D', drop_flags='BAD')

    assert flagger.isFlagged(flags, field)[[3, 5, 13, 14]].all()

