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
    flagPattern
)

from test.common import initData, TESTFLAGGER

import pandas as pd
import dios



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
    assert (flagged == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("method", ['wavelet', 'dtw'])
@pytest.mark.parametrize("pattern", [pytest.lazy_fixture("course_pattern_1"),
                                     pytest.lazy_fixture("course_pattern_2"),] ,)
                            #     pytest.lazy_fixture("course_3"),
                            #     pytest.lazy_fixture("course_4"), ],
                                           #  )
def test_flagPattern(course_test, flagger, method, pattern):
    pattern_data, dict_pattern = pattern()

    # testing the same pattern sampled at different frequencies
    if pattern_data.columns == "pattern1":
        test_data, *_ = course_test(freq="10 min")
        test_data['pattern_data'] = pattern_data.to_df()
        flagger = flagger.initFlags(test_data)
        data, flagger = flagPattern(test_data, "data", flagger, reference_field="pattern_data", partition_freq="1 H", method=method)
        assert flagger.isFlagged("data")[dict_pattern["pattern_1"]].all()
    if pattern_data.columns == "pattern2":
        test_data, *_ = course_test(freq="1 H")
        test_data['pattern_data'] = pattern_data.to_df()
        flagger = flagger.initFlags(test_data)
        data, flagger = flagPattern(test_data, "data", flagger, reference_field="pattern_data", partition_freq="days", method=method)
        assert flagger.isFlagged("data")[dict_pattern["pattern_2"]].all()







@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSesonalRange(data, field, flagger):
    # prepare
    data.iloc[::2] = 0
    data.iloc[1::2] = 50
    nyears = len(data[field].index.year.unique())

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
    assert (flags_orig != flags_set).all(None)
    assert (flags_orig == flags_cleared).all(None)


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
    s = data[field].iloc[5:6]
    flagger = flagger.setFlags(field, loc=s)

    _, flagger_result = flagIsolated(data, field, flagger, group_window="1D", gap_window="2.1D")

    assert flagger_result.isFlagged(field)[slice(3, 6, 2)].all()

    data, flagger_result = flagIsolated(
        data, field, flagger_result, group_window="2D", gap_window="2.1D", continuation_range="1.1D",
    )
    assert flagger_result.isFlagged(field)[[3, 5, 13, 14]].all()
