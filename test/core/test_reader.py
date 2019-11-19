#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from saqc.core.reader import checkConfig
from saqc.core.config import Fields as F
from test.common import (
    initData,
    initMetaDict,
    initMetaString,
    TESTFLAGGER,
    TESTNODATA)


@pytest.fixture
def data():
    return initData(3)


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


def test_configReaderLineNumbers(data):
    config = f"""
    {F.VARNAME}|{F.TESTS}
    #temp1|flagAll()
    pre1|flagAll()
    pre2|flagAll()
    SM|flagAll()
    #SM|flagAll()
    SM1|flagAll()
    """
    meta_fname, meta_frame = initMetaString(config, data)
    result = meta_frame[F.LINENUMBER].tolist()
    expected = [2, 3, 4, 6]
    assert result == expected


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_configChecks(data, flagger, nodata, caplog):

    flags = flagger.initFlags(data)
    var1, var2, var3, *_ = data.columns

    tests = [
        ({F.VARNAME: var1, F.TESTS: "range(mn=0)"}, TypeError),
        ({F.VARNAME: var1, F.TESTS: "range(min=xxx)"}, TypeError),
        ({F.VARNAME: "temp2", F.TESTS: "range(min=3)"}, NameError),
        ({F.VARNAME: var3, F.TESTS: "flagNothing()"}, NameError),
        ({F.VARNAME: "", F.TESTS: "range(min=3)"}, NameError),
        ({F.VARNAME: "", F.TESTS: ""}, NameError),
        ({F.VARNAME: ""}, SyntaxError),
        ({F.TESTS: "range(min=3)"}, SyntaxError)
        ]

    for config_dict, expected in tests:
        _, config_df = initMetaDict([config_dict], data)
        with pytest.raises(expected):
            checkConfig(config_df, data, flags, flagger, nodata)