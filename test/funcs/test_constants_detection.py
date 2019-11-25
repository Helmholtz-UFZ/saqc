#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.categoricalflagger import CategoricalBaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.constants_detection import flagConstants_VarianceBased



TESTFLAGGERS = [
    CategoricalBaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.fixture(scope='module')
def constants_data():
    index = pd.date_range(start='2011-01-01 00:00:00', end='2011-01-01 03:00:00', freq='5min')
    constants_series = pd.DataFrame(dict(constants_data=np.linspace(-50, 50, index.size)), index=index)
    constants_series.iloc[5:25] = 0
    # constants_series.iloc[1000] = -100
    flag_assertion = list(range(5, 25))
    return constants_series, flag_assertion


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagConstants_VarianceBased(constants_data, flagger):
    data = constants_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagConstants_VarianceBased(data, flags, 'constants_data', flagger, plateau_window_min='1h')
    flag_result = flag_result.iloc[:, 0]
    test_sum = (flag_result[constants_data[1]] == flagger.BAD).sum()
    assert test_sum == len(constants_data[1])







