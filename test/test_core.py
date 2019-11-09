#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.funcs import register
from saqc.core.core import runner, flagNext, flagPeriod
from saqc.core.config import Fields as F
from .common import initData, initMetaDict, TESTFLAGGER


@pytest.fixture
def data():
    return initData(3)


@register("flagAll")
def flagAll(data, flags, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    return data, flagger.setFlags(flags, field, flag=flagger.BAD)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_temporalPartitioning(data, flagger):
    """
    Check if the time span in meta is respected
    """
    var1, var2, var3, *_ = data.columns
    split_date = data.index[len(data.index)//2]

    metadict = [
        {F.VARNAME: var1, "Flag": "flagAll()"},
        {F.VARNAME: var2, "Flag": "flagAll()", F.END: split_date},
        {F.VARNAME: var3, "Flag": "flagAll()", F.START: split_date},
    ]
    meta_file, meta_frame = initMetaDict(metadict, data)

    pdata, pflags = runner(meta_file, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflags.loc[flagger.isFlagged(pflags[vname]), vname]
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_positionalPartitioning(data, flagger):
    data = data.reset_index(drop=True)
    var1, var2, var3, *_ = data.columns
    split_index = int(len(data.index)//2)

    metadict = [
        {F.VARNAME: var1, "Flag": "flagAll()"},
        {F.VARNAME: var2, "Flag": "flagAll()", F.END: split_index},
        {F.VARNAME: var3, "Flag": "flagAll()", F.START: split_index},
    ]
    meta_file, meta_frame = initMetaDict(metadict, data)

    pdata, pflags = runner(meta_file, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_index, end_index = row[fields]
        fchunk = pflags.loc[flagger.isFlagged(pflags[vname]), vname]
        assert fchunk.index.min() == start_index, "different start indices"
        assert fchunk.index.max() == end_index, f"different end indices: {fchunk.index.max()} vs. {end_index}"


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingConfig(data, flagger):
    """
    Test if variables available in the dataset but not the config
    are handled correctly, i.e. are ignored
    """
    var1, var2, *_ = data.columns

    metadict = [{F.VARNAME: var1, "Flag": "flagAll()"}]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    assert var1 in pdata and var2 not in pflags


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingVariable(flagger):
    """
    Test if variables available in the config but not dataset
    are handled correctly, i.e. are ignored
    """
    data = initData(1)
    var, *_ = data.columns

    metadict = [
        {F.VARNAME: var, "Flag": "flagAll()"},
        {F.VARNAME: "empty", "Flag": "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflags = runner(metafobj, flagger, data)

    assert (pdata.columns == [var]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_assignVariable(flagger):
    """
    Test the assign keyword, a variable present in the configuration, but not
    dataset will be added to output flags
    """
    data = initData(1)
    var1, *_ = data.columns
    var2 = "empty"

    metadict = [
        {F.VARNAME: var1, F.ASSIGN: False, "Flag": "flagAll()"},
        {F.VARNAME: var2, F.ASSIGN: True,  "Flag": "flagAll()"},
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


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_dtypes(data, flagger):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    metadict = [
        {F.VARNAME: var1, "Flag": "flagAll()"},
        {F.VARNAME: var2, "Flag": "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)
    pdata, pflags = runner(metafobj, flagger, data, flags)
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagNext(flagger):
    """
    Test if the flagNext functionality works as expected

    NOTE:
    needs to move out of this module
    """
    data = initData()
    flags = flagger.initFlags(data)
    orig = flags.copy()
    var1 = 'var1'

    idx = [0, 1, 2]
    dtidx = data.index[idx]
    flags = flagger.setFlags(flags, var1, dtidx)

    n = 4
    fflags = flagNext(orig, flags, var1, flagger, flag_values=4)
    flagged = flagger.isFlagged(fflags[var1])
    ffindex = fflags[flagged].index

    expected = data.index[min(idx):max(idx)+n+1]
    assert (expected == ffindex).all()
    o = flagger.getFlags(orig).loc[expected, var1]
    f = flagger.getFlags(fflags).loc[flagged, var1]
    assert (o != f).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagPeriod(flagger):
    """
    Test if the flagNext functionality works as expected

    NOTE:
    needs to move out of this module
    """
    data = initData()
    flags = flagger.initFlags(data)
    orig = flags.copy()
    var1 = 'var1'

    idx = [0, 1, 2]
    dtidx = data.index[idx]
    flags = flagger.setFlags(flags, var1, dtidx)

    period = '4h'
    fflags = flagPeriod(orig, flags, var1, flagger, flag_period=period)
    flagged = flagger.isFlagged(fflags[var1])
    ffindex = fflags[flagged].index

    m, M = data.index[min(idx)], data.index[max(idx)] + pd.to_timedelta(period)
    expected = data.loc[m:M].index
    assert (expected == ffindex).all()
    o = flagger.getFlags(orig).loc[expected, var1]
    f = flagger.getFlags(fflags).loc[flagged, var1]
    assert (o != f).all()
