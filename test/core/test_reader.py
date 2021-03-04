#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.core.config import Fields as F
from test.common import initData, writeIO

from saqc.core.core import SaQC
from saqc.core.register import FUNC_MAP, register


@pytest.fixture
def data() -> dios.DictOfSeries:
    return initData(3)


def test_packagedConfig():

    path = Path(__file__).parents[2] / "ressources/data"

    config_path = path / "config_ci.csv"
    data_path = path / "data.csv"

    data = pd.read_csv(data_path, index_col=0, parse_dates=True,)
    saqc = SaQC(dios.DictOfSeries(data)).readConfig(config_path)
    saqc.getResult()


def test_variableRegex(data):

    header = f"{F.VARNAME};{F.TEST};{F.PLOT}"
    tests = [
        ("'.*'", data.columns),
        ("'var(1|2)'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("'var[12]'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("var[12]", ["var[12]"]),  # not quoted -> not a regex
        ('".*3"', [c for c in data.columns if c[-1] == "3"]),
    ]

    for regex, expected in tests:
        fobj = writeIO(header + "\n" + f"{regex} ; flagtools.flagDummy()")
        saqc = SaQC(data).readConfig(fobj)
        expansion = saqc._expandFields(saqc._to_call[0][0], saqc._to_call[0][2], data.columns)
        result = [s.field for s, _ in expansion]
        assert np.all(result == expected)


def test_inlineComments(data):
    """
    adresses issue #3
    """
    config = f"""
    {F.VARNAME} ; {F.TEST}       ; {F.PLOT}
    pre2        ; flagtools.flagDummy() # test ; False # test
    """
    saqc = SaQC(data).readConfig(writeIO(config))
    _, control, func = saqc._to_call[0]
    assert control.plot is False
    assert func.func == FUNC_MAP["flagtools.flagDummy"].func


def test_configReaderLineNumbers(data):
    config = f"""
    {F.VARNAME} ; {F.TEST}
    #temp1      ; flagtools.flagDummy()
    pre1        ; flagtools.flagDummy()
    pre2        ; flagtools.flagDummy()
    SM          ; flagtools.flagDummy()
    #SM         ; flagtools.flagDummy()
    # SM1       ; flagtools.flagDummy()

    SM1         ; flagtools.flagDummy()
    """
    saqc = SaQC(data).readConfig(writeIO(config))
    result = [c.lineno for _, c, _ in saqc._to_call]
    expected = [3, 4, 5, 9]
    assert result == expected


def test_configFile(data):

    # check that the reader accepts different whitespace patterns

    config = f"""
    {F.VARNAME} ; {F.TEST}

    #temp1      ; flagtools.flagDummy()
    pre1; flagtools.flagDummy()
    pre2        ;flagtools.flagDummy()
    SM          ; flagtools.flagDummy()
    #SM         ; flagtools.flagDummy()
    # SM1       ; flagtools.flagDummy()

    SM1;flagtools.flagDummy()
    """
    SaQC(data).readConfig(writeIO(config))


def test_configChecks(data):

    var1, _, var3, *_ = data.columns

    @register(masking="none")
    def flagFunc(data, field, flagger, arg, opt_arg=None, **kwargs):
        return data, flagger

    header = f"{F.VARNAME};{F.TEST}"
    tests = [
        (f"{var1};flagFunc(mn=0)", TypeError),  # bad argument name
        (f"{var1};flagFunc()", TypeError),  # not enough arguments
        (f"{var3};flagNothing()", NameError),  # unknown function
        (f"{var1}; min", TypeError),  # not a function call
    ]

    for test, expected in tests:
        fobj = writeIO(header + "\n" + test)
        with pytest.raises(expected):
            SaQC(data).readConfig(fobj).getResult()


def test_supportedArguments(data):

    # test if the following function arguments
    # are supported (i.e. parsing does not fail)

    # TODO: necessary?

    @register(masking='field')
    def func(data, field, flagger, kwarg, **kwargs):
        return data, flagger

    var1 = data.columns[0]

    header = f"{F.VARNAME};{F.TEST}"
    tests = [
        f"{var1};func(kwarg=NAN)",
        f"{var1};func(kwarg='str')",
        f"{var1};func(kwarg=5)",
        f"{var1};func(kwarg=5.5)",
        f"{var1};func(kwarg=-5)",
        f"{var1};func(kwarg=True)",
        f"{var1};func(kwarg=sum([1, 2, 3]))",
    ]

    for test in tests:
        fobj = writeIO(header + "\n" + test)
        SaQC(data).readConfig(fobj)
