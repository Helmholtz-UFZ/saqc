#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.funcs import register, flagRange
from saqc.core.core import runner
from saqc.core.config import Fields as F
from saqc.lib.plotting import plot
from test.common import initData, initMetaDict, initMetaString, TESTFLAGGER


@pytest.fixture
def data():
    return initData(3)

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
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        # {F.VARNAME: var2, F.TESTS: "flagAll()", F.END: split_date},
        # {F.VARNAME: var3, F.TESTS: "flagAll()", F.START: split_date},
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
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.END: split_index},
        {F.VARNAME: var3, F.TESTS: "flagAll()", F.START: split_index},
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

    metadict = [{F.VARNAME: var1, F.TESTS: "flagAll()"}]
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
        {F.VARNAME: var, F.TESTS: "flagAll()"},
        {F.VARNAME: "empty", F.TESTS: "flagAll()"},
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
        {F.VARNAME: var1, F.ASSIGN: False, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.ASSIGN: True,  F.TESTS: "flagAll()"},
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
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, "test": "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)
    pdata, pflags = runner(metafobj, flagger, data, flags)
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_plotting(data, flagger):
    """
    Test if the plotting code runs, does not show any plot.

    NOTE:
    This test is ignored if matplotlib is not available on the test-system
    """
    pytest.importorskip("matplotlib", reason="requires matplotlib")
    field, *_ = data.columns
    flags = flagger.initFlags(data)
    _, flagged = flagRange(data, flags, field, flagger, min=10, max=90, flag=flagger.BAD)
    _, flagged = flagRange(data, flagged, field, flagger, min=40, max=60, flag=flagger.GOOD)
    mask = flagger.getFlags(flags[field]) != flagger.getFlags(flagged[field])
    plot(data, flagged, mask, field, flagger, interactive_backend=False)


def test_configPreparation(data):
    var1, var2, var3, *_ = data.columns
    date = data.index[len(data.index)//2]

    tests = [
        {F.VARNAME: var1, F.START: date, F.TESTS: "flagAll()", F.PLOT: True},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.PLOT: False},
        {F.VARNAME: var3, F.END: date, F.TESTS: "flagAll()", F.ASSIGN: True},
        {F.VARNAME: var3, F.TESTS: "flagAll()", },
    ]

    defaults = {
        F.START: data.index.min(), F.END: data.index.max(),
        F.ASSIGN: False, F.PLOT: False, F.LINENUMBER: 1
    }

    for i, test in enumerate(tests):
        _, meta_frame = initMetaDict([test], data)
        result = dict(zip(meta_frame.columns, meta_frame.iloc[0]))
        expected = {**defaults, **test}
        assert result == expected


def test_configPreparationExcpetion(data):
    var1, var2, var3, *_ = data.columns
    date = data.index[len(data.index)//2]

    tests = [
        {},
        # {F.TESTS: "flagAll()"},
        # {F.VARNAME: var2},
        # {F.VARNAME: var3, F.END: date, F.ASSIGN: True},
    ]

    for test in tests:
        with pytest.raises(TypeError):
            initMetaDict([test], data)


def test_configReaderLineNumbers(data):
    config = f"""
    {F.VARNAME}|{F.TESTS}
    #temp1|flagAll()
    temp1|flagAll()
    temp2|flagAll()
    pre1|flagAll()
    pre2|flagAll()
    SM|flagAll()
    #SM|flagAll()
    SM1|flagAll()
    """
    meta_fname, meta_frame = initMetaString(config, data)
    result = meta_frame[F.LINENUMBER].tolist()
    expected = [2, 3, 4, 5, 6, 8]
    assert result == expected
