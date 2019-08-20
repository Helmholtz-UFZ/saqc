#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.soil_moisture_tests import flagSoilMoistureBySoilFrost, flagSoilMoistureByPrecipitationEvents

from saqc.lib.tools import getPandasData

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSoilMoistureBySoilFrost(flagger):
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 03:00:00', freq='5min')
    data = pd.DataFrame({'soil_moisture': np.linspace(0, 1, index.size),
                         'soil_temperature': np.linspace(1, -1, index.size)}, index=index)
    flags = flagger.initFlags(data)
    data, flag_result = flagSoilMoistureBySoilFrost(data, flags, 'soil_moisture', flagger, 'soil_temperature')
    flag_assertion = list(range(18, 37))
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[flag_assertion] == flagger.BAD).sum()
    assert test_sum == len(flag_assertion)

@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSoilMoisturePrecipitationEvents(flagger):
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.4.2011 00:00:00', freq='15min')
    data = pd.DataFrame({'soil_moisture': np.linspace(0, 1, index.size),
                         'precipitation': np.linspace(1, 1, index.size)}, index=index)
    data['precipitation']['1.3.2011'] = 0
    data['precipitation']['1.4.2011'] = 0
    flags = flagger.initFlags(data)
    data, flag_result = flagSoilMoistureByPrecipitationEvents(data, flags, 'soil_moisture', flagger, 'precipitation')
    flag_assertion = [288, 287]
    flag_result = getPandasData(flag_result, 0)
    test_sum = (flag_result[flag_assertion] == flagger.BAD).sum()
    assert test_sum == len(flag_assertion)


