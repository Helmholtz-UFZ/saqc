#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.funcs import register, flagRange
from saqc.core.core import runner
from saqc.core.config import Fields as F
from saqc.lib.plotting import _plot
from test.common import initData, initMetaDict, TESTFLAGGER


@pytest.fixture
def data():
    return initData(3)


@register("flagAll")
def flagAll(data, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    return data, flagger.setFlags(field=field, flag=flagger.BAD)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_temporalPartitioning(data, flagger):
    """
    Check if the time span in meta is respected
    """
    var1, var2, var3, *_ = data.columns
    split_date = data.index[len(data.index) // 2]

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.END: split_date},
        {F.VARNAME: var3, F.TESTS: "flagAll()", F.START: split_date},
    ]
    meta_file, meta_frame = initMetaDict(metadict, data)
    pdata, pflagger = runner(meta_file, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflagger.getFlags(field=vname, loc=pflagger.isFlagged(vname))
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_positionalPartitioning(data, flagger):
    data = data.reset_index(drop=True)
    var1, var2, var3, *_ = data.columns
    split_index = int(len(data.index) // 2)

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.END: split_index},
        {F.VARNAME: var3, F.TESTS: "flagAll()", F.START: split_index},
    ]
    meta_file, meta_frame = initMetaDict(metadict, data)

    pdata, pflagger = runner(meta_file, flagger, data)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_index, end_index = row[fields]
        fchunk = pflagger.getFlags(field=vname, loc=pflagger.isFlagged(vname))
        assert fchunk.index.min() == start_index, "different start indices"
        assert (
            fchunk.index.max() == end_index
        ), f"different end indices: {fchunk.index.max()} vs. {end_index}"


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingConfig(data, flagger):
    """
    Test if variables available in the dataset but not the config
    are handled correctly, i.e. are ignored
    """
    var1, var2, *_ = data.columns

    metadict = [{F.VARNAME: var1, F.TESTS: "flagAll()"}]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflagger = runner(metafobj, flagger, data)

    assert var1 in pdata and var2 not in pflagger.getFlags()


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
    with pytest.raises(NameError):
        runner(metafobj, flagger, data)


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
        {F.VARNAME: var2, F.ASSIGN: True, F.TESTS: "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflagger = runner(metafobj, flagger, data)
    pflags = pflagger.getFlags()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert (cols == [var1, var2]).all()
        assert pflagger.isFlagged(var2).any()
    else:
        assert (pflags.columns == [var1, var2]).all()
        assert pflagger.isFlagged(var2).any()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_dtypes(data, flagger):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    var1, var2, *_ = data.columns

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)
    pdata, pflagger = runner(metafobj, flagger, data, flags)
    pflags = pflagger.getFlags()
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.skip(reason="not ported yet")
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_plotting(data, flagger):
    """
    Test if the plotting code runs, does not show any plot.

    NOTE:
    This test is ignored if matplotlib is not available on the test-system
    """
    pytest.importorskip("matplotlib", reason="requires matplotlib")
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    _, flagger_range = flagRange(data, field, flagger, min=10, max=90, flag=flagger.BAD)
    _, flagger_range = flagRange(
        data, field, flagger_range, min=40, max=60, flag=flagger.GOOD
    )
    mask = flagger.getFlags(field) != flagger_range.getFlags(field)
    plot(data, mask, field, flagger, interactive_backend=False)
