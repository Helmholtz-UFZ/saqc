#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from saqc import BAD, UNFLAGGED, DictOfSeries, Flags, SaQC
from saqc.core import register
from saqc.core.flags import initFlagsLike
from saqc.funcs.generic import compileGeneric, execGeneric
from saqc.parsing.reader import _ConfigReader
from tests.common import initData


@pytest.fixture
def data():
    return initData()


@pytest.fixture
def data_diff():
    data = initData(cols=3)
    col0 = data[data.columns[0]]
    col1 = data[data.columns[1]]
    mid = len(col0) // 2
    offset = len(col0) // 8
    return DictOfSeries(
        {
            data.columns[0]: col0.iloc[: mid + offset],
            data.columns[1]: col1.iloc[mid - offset :],
        }
    )


@pytest.mark.parametrize(
    "expr, error",
    [
        ("range(x=5", SyntaxError),
        ("rangex=5)", SyntaxError),
        ("range[x=5]", SyntaxError),
        ("range{x=5}", SyntaxError),
        ("x.groupby(by='y')", AttributeError),
        ("lambda x: x * 2", TypeError),
        ("x + x.loc['2012-01-01']", TypeError),
        ("x.mean({'a': 1})", TypeError),
    ],
)
def test_invalid_constructs(expr, error):
    with pytest.raises(error):
        compileGeneric(expr)


@pytest.mark.parametrize(
    "fields,expr,expected",
    [
        (["var1"], "x > 100", 'data["var1"] > 100'),
        (["var2"], "10 >= y", '10 >= data["var2"]'),
        (["var2"], "y < 100", 'data["var2"] < 100'),
        (["var1", "var2"], "x <= y", 'data["var1"] <= data["var2"]'),
        (["var1", "var2"], "x == y", 'data["var1"] == data["var2"]'),
        (["var1", "var2"], "x != y", 'data["var1"] != data["var2"]'),
    ],
)
def test_comparisonOperators(data, fields, expr, expected):
    expected = eval(expected)
    flags = initFlagsLike(data)
    func = compileGeneric(expr)
    result = execGeneric(Flags({f: flags[f] for f in fields}), data[fields], func)
    assert (result == expected).all(axis=None)


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("var1 + 100 > 110", 'data["var1"] + 100 > 110'),
        ("var1 - 100 > 0", 'data["var1"] - 100 > 0'),
        ("var1 * 100 > 200", 'data["var1"] * 100 > 200'),
        ("var1 / 100 > .1", 'data["var1"] / 100 > 0.1'),
        ("var1 % 2 == 1", 'data["var1"] % 2 == 1'),
        ("var1 ** 2 == 0", 'data["var1"]**2 == 0'),
    ],
)
def test_arithmeticOperators(data, expr, expected):
    expected = eval(expected)
    flags = Flags({"var1": pd.Series(UNFLAGGED, index=data["var1"].index)})
    func = compileGeneric(expr)
    result = execGeneric(flags, data["var1"], func)
    assert (result == expected).all(axis=None)


def test_nonReduncingBuiltins(data):
    var1, *_ = data.columns
    data = data[var1].iloc[1:10]
    flags = Flags({var1: pd.Series(UNFLAGGED, index=data.index)})

    tests = [
        ("abs(x)", np.abs(data)),
        ("log(x)", np.log(data)),
        ("exp(x)", np.exp(data)),
    ]

    for test, expected in tests:
        func = compileGeneric(test)
        result = execGeneric(flags, data, func)
        assert (result == expected).all(axis=None)


def test_bitOps(data):
    var1, var2, *_ = data.columns
    flags = initFlagsLike(data)

    tests = [
        ([var1], "~(x > mean(x))", ~(data[var1] > np.nanmean(data[var1]))),
        ([var1], "(x <= 0) | (0 < x)", (data[var1] <= 0) | (0 < data[var1])),
        ([var1, var2], "(y>= 0) & (0 > x)", (data[var2] >= 0) & (0 > data[var1])),
    ]

    for field, test, expected in tests:
        func = compileGeneric(test)
        result = execGeneric(Flags({f: flags[f] for f in field}), data[field], func)
        assert (result == expected).all(axis=None)


def test_variableAssignments(data):
    config = """
    varname ; test
    dummy1  ; processGeneric(field=["var1", "var2"], func=x + y)
    dummy2  ; flagGeneric(field=["var1", "var2"], func=x + y > 0)
    """

    cr = _ConfigReader(data)
    saqc = cr.readString(config).run()

    expected_columns = set(data.columns) | {"dummy1", "dummy2"}
    assert set(saqc.data.columns) == expected_columns
    assert set(saqc.flags.columns) == expected_columns


def test_processExistingTarget(data):
    config = """
    varname ; test
    var2   ; flagNAN()
    var2   ; processGeneric(func=y - 1)
    """

    cr = _ConfigReader(data)
    saqc = cr.readString(config).run()
    assert (saqc._data["var2"] == data["var2"] - 1).all()
    assert len(saqc._flags.history["var2"]) == 2
    assert saqc._flags.history["var2"].hist[0].isna().all()
    assert saqc._flags.history["var2"].hist[1].isna().all()


def test_flagTargetExisting(data):
    config = """
    varname ; test
    dummy   ; processGeneric(field="var1", func=x < 1)
    dummy   ; processGeneric(field="var2", func=y >1)
    """

    cr = _ConfigReader(data)
    saqc = cr.readString(config).run()
    assert len(saqc.data["dummy"]) == len(saqc.flags["dummy"])


def test_processTargetExistingFail(data_diff):
    config = """
    varname ; test
    dummy   ; processGeneric(field="var1", func=x + 1)
    dummy   ; processGeneric(field="var2", func=y - 1)
    """

    cr = _ConfigReader(data_diff).readString(config)
    with pytest.raises(ValueError):
        cr.run()


def test_callableArgumentsUnary(data):
    window = 5

    @register(mask=["field"], demask=["field"], squeeze=["field"], mode="processing")
    def testFuncUnary(saqc, field, func, **kwargs):
        value = saqc._data[field].rolling(window=window).apply(func)
        saqc._data[field] = value
        return saqc

    var = data.columns[0]

    # we slice the data, because the test is very slow otherwise
    data[var] = data[var].iloc[:100]

    config = f"""
    varname ; test
    {var}   ; testFuncUnary(func={{0}})
    """

    tests = [
        # ("sum", np.nansum),
        ("std(exp(x))", lambda x: np.std(np.exp(x))),
    ]

    for name, func in tests:
        cr = _ConfigReader(data).readString(config.format(name))
        result_config = cr.run().data
        result_api = SaQC(data).testFuncUnary(var, func=func).data
        expected = data[var].rolling(window=window).apply(func)
        assert (result_config[var].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var].dropna() == expected.dropna()).all(axis=None)


def test_callableArgumentsBinary(data):
    var1, var2 = data.columns[:2]

    @register(mask=["field"], demask=["field"], squeeze=["field"], mode="processing")
    def testFuncBinary(saqc, field, func, **kwargs):
        saqc._data[field] = func(data[var1], data[var2])
        return saqc

    config = f"""
    varname ; test
    {var1}  ; testFuncBinary(func={{0}})
    """

    tests = [
        ("x + y", lambda x, y: x + y),
        ("y - (x * 2)", lambda y, x: y - (x * 2)),
    ]

    for name, func in tests:
        cr = _ConfigReader(data).readString(config.format(name))
        result_config = cr.run().data
        result_api = SaQC(data).testFuncBinary(var1, func=func).data
        expected = func(data[var1], data[var2])
        assert (result_config[var1].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var1].dropna() == expected.dropna()).all(axis=None)


def test_isflagged(data):
    var1, var2, *_ = data.columns
    flags = initFlagsLike(data)
    flags[data[var1].index[::2], var1] = BAD

    tests = [
        ([var1], "isflagged(x)", flags[var1] > UNFLAGGED),
        ([var1], "isflagged(x)", flags[var1] >= BAD),
        ([var2], "~isflagged(x)", flags[var2] == UNFLAGGED),
        (
            [var1, var2],
            "~(x > 999) & (~isflagged(y))",
            ~(data[var1] > 999) & (flags[var2] == UNFLAGGED),
        ),
    ]

    for field, test, expected in tests:
        func = compileGeneric(test)
        result = execGeneric(Flags({f: flags[f] for f in field}), data[field], func)
        assert (result == expected).all(axis=None)


@pytest.mark.parametrize(
    "func_string, expected, n_args",
    [
        ("x.rolling(window=10).mean()", lambda x: x.rolling(window=10).mean(), 1),
        ("x.resample('D').sum()", lambda x: x.resample("D").sum(), 1),
        ("x / x.var()", lambda x: x / x.var(), 1),
        ("x / y.var()", lambda x, y: x / y.var(), 2),
        ("y / z.cummax()", lambda x, y: x / y.cummax(), 2),
    ],
)
def test_pandas_methods(data, func_string, expected, n_args):

    test_data = data[data.columns[:n_args]]
    func = compileGeneric(func_string)
    result = func(*test_data.values())
    assert result.equals(expected(*test_data.values()))
