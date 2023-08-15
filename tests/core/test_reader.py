#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pytest

from saqc.core import DictOfSeries, Flags, SaQC, flagging
from saqc.exceptions import ParsingError
from saqc.parsing.environ import ENVIRONMENT
from saqc.parsing.reader import _ConfigReader
from tests.common import initData


@pytest.fixture
def data() -> DictOfSeries:
    return initData(3)


def getTestedVariables(flags: Flags, test: str):
    out = []
    for col in flags.columns:
        for m in flags.history[col].meta:
            if m["func"] == test:
                out.append(col)
    return out


def test_variableRegex(data):
    header = f"varname;test"
    function = "flagDummy"
    tests = [
        ("'.*'", data.columns),
        ("'var(1|2)'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ("'var[12]'", [c for c in data.columns if c[-1] in ("1", "2")]),
        ('".*3"', [c for c in data.columns if c[-1] == "3"]),
    ]

    for regex, expected in tests:
        cr = _ConfigReader(data)
        cr.readString(header + "\n" + f"{regex} ; {function}()")
        saqc = cr.run()
        result = getTestedVariables(saqc._flags, function)
        assert np.all(result == expected)

    tests = [
        ("var[12]", []),  # not quoted -> not a regex
    ]
    for regex, expected in tests:
        cr = _ConfigReader(data=data)
        cr.readString(header + "\n" + f"{regex} ; {function}()")
        with pytest.warns(RuntimeWarning):
            saqc = cr.run()
        result = getTestedVariables(saqc._flags, function)
        assert np.all(result == expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inlineComments(data):
    """
    adresses issue #3
    """
    config = f"""
    varname ; test
    var1    ; flagDummy() # test
    """

    saqc = _ConfigReader(data).readString(config).run()
    func = saqc._flags.history["var1"].meta[0]["func"]
    assert func == "flagDummy"


def test_configReaderLineNumbers():
    config = f"""
    varname ; test
    #temp1      ; flagDummy()
    pre1        ; flagDummy()
    pre2        ; flagDummy()
    SM          ; flagDummy()
    #SM         ; flagDummy()
    # SM1       ; flagDummy()

    SM1         ; flagDummy()
    """
    planned = _ConfigReader().readString(config)
    expected = [4, 5, 6, 10]
    assert (planned.config.index == expected).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_configFile(data):
    # check that the reader accepts different whitespace patterns

    config = f"""
    varname ; test

    #temp1      ; flagDummy()
    pre1; flagDummy()
    pre2        ;flagDummy()
    SM          ; flagDummy()
    #SM         ; flagDummy()
    # SM1       ; flagDummy()

    SM1;flagDummy()
    """
    c = _ConfigReader().readString(config).config
    assert len(c) == 4


@pytest.mark.parametrize(
    "test, expected",
    [
        (f"var1; min", ParsingError),  # not a function call
        (f"var3; flagNothing()", NameError),  # unknown function
        (f"var1; flagFunc(mn=0)", TypeError),  # bad argument name
        (f"var1; flagFunc()", TypeError),  # not enough arguments
    ],
)
def test_configChecks(data, test, expected):
    @flagging()
    def flagFunc(data, field, flags, arg, opt_arg=None, **kwargs):
        flags[:, field] = np.nan
        return data, flags

    header = f"varname;test"
    cr = _ConfigReader(data).readString(header + "\n" + test)
    with pytest.raises(expected):
        cr.run()


@pytest.mark.parametrize(
    "kwarg", ["NAN", "'a string'", "5", "5.5", "-5", "True", "sum([1, 2, 3])"]
)
def test_supportedArguments(data, kwarg):
    # test if the following function arguments
    # are supported (i.e. parsing does not fail)

    # TODO: necessary?

    @flagging()
    def func(saqc, field, kwarg, **kwargs):
        saqc._flags[:, field] = np.nan
        return saqc

    var1 = data.columns[0]
    conf = f"varname;test" + "\n" + f"{var1};func(kwarg={kwarg})"
    _ConfigReader(data).readString(conf).run()


@pytest.mark.parametrize(
    "func_string", [k for k, v in ENVIRONMENT.items() if callable(v)]
)
def test_funtionArguments(data, func_string):
    @flagging()
    def testFunction(saqc, field, func, **kwargs):
        assert func is ENVIRONMENT[func_string]
        return saqc

    config = f"""
    varname ; test
    {data.columns[0]} ; testFunction(func={func_string})
    {data.columns[0]} ; testFunction(func="{func_string}")
    """
    cr = _ConfigReader(data)
    cr.readString(config)
    cr.run()
