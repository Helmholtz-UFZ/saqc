import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from saqc.funcs.machine_learning import flagML

from saqc.flagger.categoricalflagger import CategoricalBaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger
from saqc.flagger.continuousflagger import ContinuousBaseFlagger

TESTFLAGGERS = [
    CategoricalBaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger(),
    ContinuousBaseFlagger()]

@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagML(flagger):
    ### CREATE MWE DATA
    data = pd.read_feather("ressources/machine_learning/data/soil_moisture_mwe.feather")
    data = data.set_index(pd.DatetimeIndex(data.Time))
    flags_raw = data[["SM1_Flag","SM2_Flag","SM3_Flag"]]
    flags_raw.columns = ["SM1","SM2","SM3"]

    # masks for flag preparation
    mask_bad = flags_raw.isin(['Auto:BattV','Auto:Range','Auto:Spike'])
    mask_unflagged = flags_raw.isin(['Manual'])
    mask_good = flags_raw.isin(['OK'])

    field = "SM2"

    # prepare flagsframe
    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags,field,loc=mask_bad[field])
    flags = flagger.setFlags(flags,field,loc=mask_unflagged[field],flag=flagger.UNFLAGGED)
    flags = flagger.setFlags(flags,field,loc=mask_good[field],flag=flagger.GOOD)

    references = ["Temp2","BattV"]
    window_values = 20
    window_flags = 20
    groupvar = 0.2
    modelname="testmodel"
    path = "ressources/machine_learning/models/"+modelname+"_"+str(groupvar)+".pkl"

    outdat, outflags = flagML(data,flags,field, flagger, references, window_values, window_flags, path)

    # compare
    #assert resulting no of bad flags
    badflags = flagger.isFlagged(outflags,field)
    assert(badflags.sum()==10447)#assert

    # Have the right values been flagged?
    checkdates = pd.DatetimeIndex(['2014-08-05 23:03:59', '2014-08-06 01:35:44',
               '2014-08-06 01:50:54', '2014-08-06 02:06:05','2014-08-06 02:21:15', '2014-08-06 04:22:38',
               '2014-08-06 04:37:49', '2014-08-06 04:52:59'])
    assert(badflags[checkdates].all())

    # IN CASE OF FUTURE changes
    # Get indices of values that were flagged
    #wasmanual = flags_raw[field]=="Manual"
    #a = badflags & wasmanual
    #data.loc[a,:].index
