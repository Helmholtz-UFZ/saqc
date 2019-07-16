#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.core.core import runner, flagNext, flagPeriod, prepareMeta, readMeta
from saqc.core.config import Fields as F
from saqc.core.config import Params as P
from saqc.flagger.simpleflagger import SimpleFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.positionalflagger import PositionalFlagger
from .common import initData, initMeta, initMetaDict


TESTFLAGGERS = [
    SimpleFlagger(),
    DmpFlagger(),
    # PositionalFlagger()
]


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_positionalPartitioning(flagger):
    data = initData(3).reset_index(drop=True)
    var1, var2, var3, *_ = data.columns
    split_index = int(len(data.index)//2)

    metadict = [
        {F.VARNAME: var1, "Flag": "range, {min: -2, max: -1}"},
        {F.VARNAME: var2, "Flag": "generic, {func: this <= sum(this)}", F.END: split_index},
        {F.VARNAME: var3, "Flag": "generic, {func: this <= sum(this)}", F.START: split_index},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta.iterrows():
        vname, start_index, end_index = row[fields]
        fchunk = pflags.loc[flagger.isFlagged(pflags[vname]), vname]
        assert fchunk.index.min() == start_index, "different start indices"
        assert fchunk.index.max() + 1 == end_index, f"different end indices: {fchunk.index.max()} vs. {end_index}"


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_temporalPartitioning(flagger):
    """
    Check if the time span in meta is respected
    """
    data = initData(3)
    var1, var2, var3, *_ = data.columns
    split_date = data.index[len(data.index)//2]

    metadict = [
        {F.VARNAME: var1, "Flag": "range, {min: -2, max: -1}"},
        {F.VARNAME: var2, "Flag": "generic, {func: this <= sum(this)}", F.END: split_date},
        {F.VARNAME: var3, "Flag": "generic, {func: this <= sum(this)}", F.START: split_date},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflags.loc[flagger.isFlagged(pflags[vname]), vname]
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_missingConfig(flagger):
    """
    Test if variables available in the dataset but not the config
    are handled correctly, i.e. are ignored
    """
    data = initData(2)
    var1, var2, *_ = data.columns

    metadict = [{F.VARNAME: var1, "Flag": "range, {min: -9999, max: 9999}"}]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    assert var1 in pdata and var2 not in pflags


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_missingVariable(flagger):
    """
    Test if variables available in the config but not dataset
    are handled correctly, i.e. are ignored
    """
    data = initData(1)
    var, *_ = data.columns

    metadict = [
        {F.VARNAME: var, "Flag": "range, {min: -9999, max: 9999}"},
        {F.VARNAME: "empty", "Flag": "range, {min: -9999, max: 9999}"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    assert (pdata.columns == [var]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_assignVariable(flagger):
    """
    Test the assign keyword, a variable present in the configuration, but not
    dataset will be added to output flags
    """
    data = initData(1)
    var1, *_ = data.columns
    var2 = "empty"

    metadict = [
        {F.VARNAME: var1, F.ASSIGN: False, "Flag": "range, {min: 9999, max: -99999}"},
        {F.VARNAME: var2, F.ASSIGN: True,  "Flag": f"generic, {{func: isflagged({var1})}}"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = (pflags
                .columns.get_level_values(0)
                .drop_duplicates())
        assert (cols == [var1, var2]).all()
        assert flagger.isFlagged(pflags[var2]).any()
    else:
        assert (pflags.columns == [var1, var2]).all()
        assert flagger.isFlagged(pflags[var2]).any()


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_dtypes(flagger):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    data = initData(1)
    flags = flagger.initFlags(data)
    var, *_ = data.columns

    metadict = [
        {F.VARNAME: var, "Flag": f"generic, {{func: this > {len(data)//2}, {P.FLAGVALUES}: 4}}"},
    ]
    metafobj, meta = initMetaDict(metadict, data)
    pdata, pflags = runner(metafobj, flagger, data, flags)
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_flagNext(flagger):
    """
    Test if the flagNext functionality works as expected
    """
    data = initData().iloc[:, 1]
    flags = flagger.initFlags(data)

    idx = [0, 1, 2]
    flags.iloc[idx] = flagger.setFlag(flags.iloc[idx])

    n = 4
    fflags = flagNext(flagger, flags.copy(), flag_values=4)
    result_idx = np.unique(np.where(flagger.isFlagged(fflags))[0])
    expected_idx = np.arange(min(idx), max(idx) + n + 1)
    assert (result_idx == expected_idx).all()


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_flagPeriod(flagger):
    """
    Test if the flagNext functionality works as expected
    """
    data = initData().iloc[:, 1]
    flags = flagger.initFlags(data)

    idx = [0, 1, 2]
    flags.iloc[idx] = flagger.setFlag(flags.iloc[idx])

    period = '4h'
    flags = flagPeriod(flagger, flags.copy(), flag_period=period)
    expected_dates = set(flags[flagger.isFlagged(flags)].index)

    tdelta = pd.to_timedelta(period)
    dates = set()
    for start in flags.index[idx]:
        stop = start + tdelta
        dates = dates | set(flags[start:stop].index)

    assert expected_dates == dates
