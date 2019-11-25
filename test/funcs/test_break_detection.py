#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.categoricalflagger import CategoricalBaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.break_detection import flagBreaks_SpektrumBased


TESTFLAGGERS = [
    CategoricalBaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.fixture(scope='module')
def break_data():
    index = pd.date_range(start='2011-01-01 00:00:00', end='2011-01-02 03:00:00', freq='5min')
    break_series = pd.DataFrame(dict(break_data=np.linspace(0, 1, index.size)), index=index)
    break_series.iloc[5:15] += 100
    flag_assertion = [5, 15]
    return break_series, flag_assertion


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagBreaks_SpektrumBased(break_data, flagger):
    data = break_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagBreaks_SpektrumBased(data, flags, 'break_data', flagger)
    flag_result = flag_result.iloc[:, 0]
    test_sum = (flag_result[break_data[1]] == flagger.BAD).sum()
    assert test_sum == len(break_data[1])


if __name__ == "__main__":
    flagger = DmpFlagger()
    data = break_data()[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagBreaks_SpektrumBased(data, flags, 'break_data', flagger)



