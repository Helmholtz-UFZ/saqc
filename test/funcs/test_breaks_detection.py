#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from saqc.funcs.breaks_detection import breaks_flagSpektrumBased
from test.common import TESTFLAGGER, initData


@pytest.fixture
def data():
    return initData(1, start_date="2011-01-01 00:00:00", end_date="2011-01-02 03:00:00", freq="5min")


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_breaks_flagSpektrumBased(data, flagger):
    field, *_ = data.columns
    data.iloc[5:15] += 100
    break_positions = [5, 15]
    flagger = flagger.initFlags(data)
    data, flagger_result = breaks_flagSpektrumBased(data, field, flagger)
    flag_result = flagger_result.getFlags(field)
    test_sum = (flag_result[break_positions] == flagger.BAD).sum()
    assert test_sum == len(break_positions)
