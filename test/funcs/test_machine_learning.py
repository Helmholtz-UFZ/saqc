import pytest

import pandas as pd

from saqc.funcs.machine_learning import flagML

from test.common import TESTFLAGGER


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagML(flagger):
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

    outdat, outflagger = flagML(
        data, field, flagger, references, window_values, window_flags, path
    )

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

    # IN CASE OF FUTURE changes
    # Get indices of values that were flagged
    # wasmanual = flags_raw[field]=="Manual"
    # a = badflags & wasmanual
    # data.loc[a,:].index
