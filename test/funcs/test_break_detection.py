#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.break_detection import flagBreaks_SpektrumBased

from saqc.lib.tools import getPandasData

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.fixture(scope='module')
def break_data():
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 00:03:00', freq='5min')
    break_series = pd.Series(np.linspace(0, 1, index.size), index=index, name='break_data')
    break_series.iloc[5:15] = +100
    flag_assertion = list(range(5, 25))
    return break_series, flag_assertion


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagBreaks_SpektrumBased(break_data, flagger):
    data = break_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagBreaks_SpektrumBased(data, flags, 'break_data', flagger, plateau_window_min='1h')
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[break_data[1]] == flagger.BAD).sum()
    assert test_sum == len(break_data[1])






