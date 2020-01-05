#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.funcs.soil_moisture_tests import (
    flagSoilMoistureBySoilFrost,
    flagSoilMoistureByPrecipitationEvents,
    flagSoilMoistureConstant
)

from test.common import TESTFLAGGER, initData


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSoilMoistureBySoilFrost(flagger):
    index = pd.date_range(
        start="2011-01-01 00:00:00", end="2011-01-01 03:00:00", freq="5min"
    )
    data = pd.DataFrame(
        {
            "soil_moisture": np.linspace(0, 1, index.size),
            "soil_temperature": np.linspace(1, -1, index.size),
        },
        index=index,
    )
    flagger = flagger.initFlags(data)
    data, flagger_result = flagSoilMoistureBySoilFrost(
        data, "soil_moisture", flagger, "soil_temperature"
    )
    flag_assertion = np.arange(19, 37)
    flag_result = flagger_result.getFlags("soil_moisture")
    assert (flag_result[flag_assertion] == flagger.BAD).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSoilMoisturePrecipitationEvents(flagger):
    index = pd.date_range(
        start="2011-01-01 00:00:00", end="2011-01-04 00:00:00", freq="15min"
    )
    data = pd.DataFrame(
        {
            "soil_moisture": np.linspace(0, 1, index.size),
            "precipitation": np.linspace(1, 1, index.size),
        },
        index=index,
    )
    data["precipitation"]["2011-01-03"] = 0
    data["precipitation"]["2011-01-04"] = 0
    flagger = flagger.initFlags(data)
    data, flag_result = flagSoilMoistureByPrecipitationEvents(
        data, "soil_moisture", flagger, "precipitation"
    )
    flag_assertion = [288, 287]
    flag_result = flag_result.getFlags("soil_moisture")
    test_sum = (flag_result[flag_assertion] == flagger.BAD).sum()
    assert test_sum == len(flag_assertion)

@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSoilMoistureConstants(flagger):

    data = initData(
        1, start_date="2011-01-01 00:00:00", end_date="2011-01-02 00:00:00", freq="5min"
    )
    data.iloc[5:25] = 0
    data.iloc[100:120] = data.max()[0]
    field = data.columns[0]
    flagger = flagger.initFlags(data)
    data, flagger = flagSoilMoistureConstant(
        data, field, flagger, window='1h', precipitation_window='1h')

    assert ~(flagger.isFlagged()[5:25]).all()[0]
    assert (flagger.isFlagged()[100:120]).all()[0]
