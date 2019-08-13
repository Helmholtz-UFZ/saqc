#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.funcs.spike_detection import flagSpikes_SpektrumBased


# @pytest.fixture(scope='module')
def spiky_data():
    index = pd.date_range(start='1.1.2011', end='2.1.2011', freq='5min')
    spiky_series = pd.Series(np.linspace(1, 2, index.size), index=index, name='spiky_data')
    spiky_series.iloc[100] = 100
    spiky_series.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return spiky_series, flag_assertion


def test_flagSpikes_SpektrumBased(spiky_data):
    flagger = BaseFlagger(['NIL', 'GOOD', 'BAD'])
    data = spiky_data[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagSpikes_SpektrumBased(data, flags, 'spiky_data', flagger)

if __name__ == '__main__':
    flagger = BaseFlagger(['NIL', 'GOOD', 'BAD'])
    data = spiky_data()[0]
    flags = flagger.initFlags(data)
    data, flag_result = flagSpikes_SpektrumBased(data, flags, 'spiky_data', flagger, diff_method='savgol')