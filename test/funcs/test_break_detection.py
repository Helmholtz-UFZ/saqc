#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from saqc.funcs.break_detection import flagBreaksSpektrumBased
from test.common import TESTFLAGGER, initData


@pytest.fixture
def data():
    return initData(
        1, start_date="2011-01-01 00:00:00", end_date="2011-01-02 03:00:00", freq="5min"
    )


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagBreaks_SpektrumBased(data, flagger):
    field, *_ = data.columns
    data.iloc[5:15] += 100
    break_positions = [5, 15]
    flagger = flagger.initFlags(data)
    data, flagger_result = flagBreaksSpektrumBased(data, field, flagger)
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[break_positions] == flagger.BAD).sum()
    assert test_sum == len(break_positions)
