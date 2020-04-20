#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np
import pandas as pd

from saqc.funcs import register, flagRange
from saqc.core.core import run
from saqc.core.config import Fields as F
from saqc.lib.plotting import _plot
from test.common import initData, initMetaDict, TESTFLAGGER


# no logging output needed here
# -> can this be configured on the test runner level?
logging.disable(logging.CRITICAL)


OPTIONAL = [False, True]


@register()
def flagAll(data, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    return data, flagger.setFlags(field=field, flag=flagger.BAD)


@pytest.fixture
def data():
    return initData(3)


def _initFlags(flagger, data, optional):
    return None
    if optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


@pytest.fixture
def flags(flagger, data, optional):
    if not optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


# NOTE: there is a lot of pytest magic involved:
#       the parametrize parameters are implicitly available
#       within the used fixtures, that is why we need the optional
#       parametrization without actually using it in the
#       function
@pytest.mark.skip(reason="test slicing support is currently disabled")
@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("optional", OPTIONAL)
def test_temporalPartitioning(data, flagger, flags):
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
    pdata, pflagger = run(meta_file, flagger, data, flags=flags)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflagger.getFlags(field=vname, loc=pflagger.isFlagged(vname))
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


@pytest.mark.skip(reason="test slicing support is currently disabled")
@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("optional", OPTIONAL)
def test_positionalPartitioning(data, flagger, flags):
    data = data.reset_index(drop=True)
    if flags is not None:
        flags = flags.reset_index(drop=True)
    var1, var2, var3, *_ = data.columns
    split_index = int(len(data.index) // 2)

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.END: split_index},
        {F.VARNAME: var3, F.TESTS: "flagAll()", F.START: split_index},
    ]
    meta_file, meta_frame = initMetaDict(metadict, data)

    pdata, pflagger = run(meta_file, flagger, data, flags=flags)

    fields = [F.VARNAME, F.START, F.END]
    for _, row in meta_frame.iterrows():
        vname, start_index, end_index = row[fields]
        fchunk = pflagger.getFlags(field=vname, loc=pflagger.isFlagged(vname))
        assert fchunk.index.min() == start_index, "different start indices"
        assert fchunk.index.max() == end_index, f"different end indices: {fchunk.index.max()} vs. {end_index}"


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_errorHandling(data, flagger):
    @register()
    def raisingFunc(data, fielf, flagger, **kwargs):
        raise TypeError

    var1, *_ = data.columns

    metadict = [
        {F.VARNAME: var1, F.TESTS: "raisingFunc()"},
    ]

    tests = ["ignore", "warn"]

    for policy in tests:
        # NOTE: should not fail, that's all we are testing here
        metafobj, _ = initMetaDict(metadict, data)
        run(metafobj, flagger, data, error_policy=policy)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_duplicatedVariable(flagger):
    data = initData(1)
    var1, *_ = data.columns

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflagger = run(metafobj, flagger, data)
    pflags = pflagger.getFlags()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert np.all(cols == [var1])
    else:
        assert (pflags.columns == [var1]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_assignVariable(flagger):
    """
    test implicit assignments
    """
    data = initData(1)
    var1, *_ = data.columns
    var2 = "empty"

    metadict = [
        {F.VARNAME: var1, F.TESTS: "flagAll()"},
        {F.VARNAME: var2, F.TESTS: "flagAll()"},
    ]
    metafobj, meta = initMetaDict(metadict, data)

    pdata, pflagger = run(metafobj, flagger, data)
    pflags = pflagger.getFlags()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert (cols == [var1, var2]).all()
        assert pflagger.isFlagged(var2).any()
    else:
        assert (pflags.columns == [var1, var2]).all()
        assert pflagger.isFlagged(var2).any()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flagger, flags):
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
    pdata, pflagger = run(metafobj, flagger, data, flags=flags)
    pflags = pflagger.getFlags()
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
    flagger = flagger.initFlags(data)
    _, flagger_range = flagRange(data, field, flagger, min=10, max=90, flag=flagger.BAD)
    _, flagger_range = flagRange(data, field, flagger_range, min=40, max=60, flag=flagger.GOOD)
    mask = flagger.getFlags(field) != flagger_range.getFlags(field)
    _plot(data, mask, field, flagger, interactive_backend=False)
