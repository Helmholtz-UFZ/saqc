#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.funcs.soil_moisture_tests import sm_flagFrost, sm_flagPrecipitation, sm_flagConstants, sm_flagRandomForest

from test.common import TESTFLAGGER, initData


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_sm_flagFrost(flagger):
    index = pd.date_range(start="2011-01-01 00:00:00", end="2011-01-01 03:00:00", freq="5min")
    data = pd.DataFrame(
        {"soil_moisture": np.linspace(0, 1, index.size), "soil_temperature": np.linspace(1, -1, index.size),},
        index=index,
    )
    flagger = flagger.initFlags(data)
    data, flagger_result = sm_flagFrost(data, "soil_moisture", flagger, "soil_temperature")
    flag_assertion = np.arange(19, 37)
    flag_result = flagger_result.getFlags("soil_moisture")
    assert (flag_result[flag_assertion] == flagger.BAD).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSoilMoisturePrecipitationEvents(flagger):
    index = pd.date_range(start="2011-01-01 00:00:00", end="2011-01-04 00:00:00", freq="15min")
    data = pd.DataFrame(
        {"soil_moisture": np.linspace(0, 1, index.size), "precipitation": np.linspace(1, 1, index.size),}, index=index,
    )
    data["precipitation"]["2011-01-03"] = 0
    data["precipitation"]["2011-01-04"] = 0
    flagger = flagger.initFlags(data)
    data, flag_result = sm_flagPrecipitation(data, "soil_moisture", flagger, "precipitation")
    flag_assertion = [288, 287]
    flag_result = flag_result.getFlags("soil_moisture")
    test_sum = (flag_result[flag_assertion] == flagger.BAD).sum()
    assert test_sum == len(flag_assertion)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_sm_flagConstantss(flagger):

    data = initData(1, start_date="2011-01-01 00:00:00", end_date="2011-01-02 00:00:00", freq="5min")
    data.iloc[5:25] = 0
    data.iloc[100:120] = data.max()[0]
    field = data.columns[0]
    flagger = flagger.initFlags(data)
    data, flagger = sm_flagConstants(data, field, flagger, window="1h", precipitation_window="1h")

    assert ~(flagger.isFlagged()[5:25]).all()[0]
    assert (flagger.isFlagged()[100:120]).all()[0]


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_sm_flagRandomForest(flagger):
    ### CREATE MWE DATA
    data = pd.read_feather("ressources/machine_learning/data/soil_moisture_mwe.feather")
    data = data.set_index(pd.DatetimeIndex(data.Time))
    flags_raw = data[["SM1_Flag", "SM2_Flag", "SM3_Flag"]]
    flags_raw.columns = ["SM1", "SM2", "SM3"]

    # masks for flag preparation
    mask_bad = flags_raw.isin(["Auto:BattV", "Auto:Range", "Auto:Spike"])
    mask_unflagged = flags_raw.isin(["Manual"])
    mask_good = flags_raw.isin(["OK"])

    field = "SM2"

    # prepare flagsframe
    flagger = flagger.initFlags(data)
    flagger = flagger.setFlags(field, loc=mask_bad[field])
    flagger = flagger.setFlags(field, loc=mask_unflagged[field], flag=flagger.UNFLAGGED)
    flagger = flagger.setFlags(field, loc=mask_good[field], flag=flagger.GOOD)

    references = ["Temp2", "BattV"]
    window_values = 20
    window_flags = 20
    groupvar = 0.2
    modelname = "testmodel"
    path = f"ressources/machine_learning/models/{modelname}_{groupvar}.pkl"

    outdat, outflagger = sm_flagRandomForest(data, field, flagger, references, window_values, window_flags, path)

    # compare
    # assert resulting no of bad flags
    badflags = outflagger.isFlagged(field)
    assert badflags.sum() == 10447

    # Have the right values been flagged?
    checkdates = pd.DatetimeIndex(
        [
            "2014-08-05 23:03:59",
            "2014-08-06 01:35:44",
            "2014-08-06 01:50:54",
            "2014-08-06 02:06:05",
            "2014-08-06 02:21:15",
            "2014-08-06 04:22:38",
            "2014-08-06 04:37:49",
            "2014-08-06 04:52:59",
        ]
    )
    assert badflags[checkdates].all()
