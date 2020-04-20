#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

import saqc
from saqc.core.reader import checkConfig
from saqc.core.config import Fields as F
from test.common import initData, initMetaDict, initMetaString, TESTFLAGGER, TESTNODATA, writeIO


@pytest.fixture
def data():
    return initData(3)


def test_configPreparation(data):
    var1, var2, var3, *_ = data.columns
    date = data.index[len(data.index) // 2]

    # NOTE:
    # time slicing support is currently disabled
    tests = [
        # {F.VARNAME: var1, F.START: date, F.TESTS: "flagAll()", F.PLOT: True},
        {F.VARNAME: var2, F.TESTS: "flagAll()", F.PLOT: False},
        # {F.VARNAME: var3, F.END: date, F.TESTS: "flagAll()"},
        {F.VARNAME: var3, F.TESTS: "flagAll()",},
    ]

    defaults = {
        F.START: data.index.min(),
        F.END: data.index.max(),
        F.PLOT: False,
        F.LINENUMBER: 2,
    }

    for i, test in enumerate(tests):
        _, meta_frame = initMetaDict([test], data)
        result = dict(zip(meta_frame.columns, meta_frame.iloc[0]))
        expected = {**defaults, **test}
        assert result == expected


def test_variableRegex(data):

    tests = [
        ("'.*'", data.columns),
        ("'var(1|2)'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("'var[12]'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("var[12]", ["var[12]"]),  # not quoted -> not a regex
        ('"(.*3)"', [c for c in data.columns if c[-1] == "3"]),
    ]

    for config_wc, expected in tests:
        _, config = initMetaDict([{F.VARNAME: config_wc, F.TESTS: "flagAll()"}], data)
        assert np.all(config[F.VARNAME] == expected)


def test_inlineComments(data):
    """
    adresses issue #3
    """
    config = f"""
    {F.VARNAME}|{F.TESTS}|{F.PLOT}
    pre2|flagAll() # test|False # test
    """
    _, meta_frame = initMetaString(config, data)
    assert meta_frame.loc[0, F.PLOT] == False
    assert meta_frame.loc[0, F.TESTS] == "flagAll()"


def test_configReaderLineNumbers(data):
    config = f"""
    {F.VARNAME}|{F.TESTS}
    #temp1|dummy()
    pre1|dummy()
    pre2|dummy()
    SM|dummy()
    #SM|dummy()
    # SM1|dummy()

    SM1|dummy()
    """
    meta_fname, meta_frame = initMetaString(config, data)
    result = meta_frame[F.LINENUMBER].tolist()
    expected = [3, 4, 5, 9]
    assert result == expected


def test_configMultipleTests(data):

    var = data.columns[0]

    config = f"""
    {F.VARNAME} ; test_1        ; test_2
    #-----------;---------------;--------------------------
    {var}       ; flagMissing() ; flagRange(min=10, max=60)
    """

    from saqc.flagger import SimpleFlagger
    from saqc.core.core import run
    from saqc.core.reader import readConfig, checkConfig
    from saqc.funcs.functions import flagMissing, flagRange

    flagger = SimpleFlagger().initFlags(data)
    df = checkConfig(readConfig(writeIO(config), data), data, flagger, np.nan)
    assert {"test_1", "test_2"} - set(df.columns) == set([])

    flagger_expected = SimpleFlagger().initFlags(data)
    for func, kwargs in [(flagMissing, {}), (flagRange, {"min": 10, "max": 60})]:
        data, flagger_expected = func(data, var, flagger_expected, **kwargs)
    _, flagger_result = run(writeIO(config), SimpleFlagger(), data)

    assert np.all(flagger_result.getFlags() == flagger_expected.getFlags())


def test_configFile(data):

    # check that the reader accepts different whitespace patterns

    config = f"""
    {F.VARNAME} ; {F.TESTS}
    #temp1      ; flagDummy()
    pre1; flagDummy()
    pre2        ;flagDummy()
    SM          ; flagDummy()
    #SM         ; flagDummy()
    # SM1       ; flagDummy()

    SM1;flagDummy()
    """
    saqc.run(writeIO(config), TESTFLAGGER[0], data)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_configChecks(data, flagger, nodata, caplog):

    flagger = flagger.initFlags(data)
    var1, var2, var3, *_ = data.columns

    tests = [
        ({F.VARNAME: var1, F.TESTS: "flagRange(mn=0)"}, TypeError),
        ({F.VARNAME: var3, F.TESTS: "flagNothing()"}, NameError),
        ({F.VARNAME: "", F.TESTS: "flagRange(min=3)"}, SyntaxError),
        ({F.VARNAME: var1, F.TESTS: ""}, SyntaxError),
        ({F.TESTS: "flagRange(min=3)"}, SyntaxError),
    ]

    for config_dict, expected in tests:
        _, config_df = initMetaDict([config_dict], data)
        with pytest.raises(expected):
            checkConfig(config_df, data, flagger, nodata)
