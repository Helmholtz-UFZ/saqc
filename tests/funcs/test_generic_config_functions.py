#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from saqc.core.flags import Flags
from saqc.core.reader import fromConfig
import pytest
import numpy as np
import pandas as pd
import dios

from saqc.constants import *
from saqc.core import initFlagsLike
from saqc.core.visitor import ConfigFunctionParser
from saqc.core.register import register
from saqc.funcs.generic import _execGeneric
from saqc import SaQC

from tests.common import initData, writeIO


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
    return dios.DictOfSeries(
        data={
            col0.name: col0.iloc[: mid + offset],
            col1.name: col1.iloc[mid - offset :],
        }
    )


def _compileGeneric(expr):
    tree = ast.parse(expr, mode="eval")
    _, kwargs = ConfigFunctionParser().parse(tree.body)
    return kwargs["func"]


def test_syntaxError():
    tests = [
        "range(x=5",
        "rangex=5)",
        "range[x=5]" "range{x=5}" "int->float(x=4)" "int*float(x=4)",
    ]

    for test in tests:
        with pytest.raises(SyntaxError):
            _compileGeneric(f"flag(func={test})")


def test_typeError():
    """
    test that forbidden constructs actually throw an error
    """

    # TODO: think about cases that should be forbidden
    tests = ("lambda x: x * 2",)

    for test in tests:
        with pytest.raises(TypeError):
            _compileGeneric(f"flagGeneric(func={test})")


def test_comparisonOperators(data):
    var1, var2, *_ = data.columns
    flags = initFlagsLike(data)

    tests = [
        (["var1"], "x > 100", data[var1] > 100),
        (["var2"], "10 >= y", 10 >= data[var2]),
        (["var2"], f"y < 100", data[var2] < 100),
        (["var1", "var2"], "x <= y", data[var1] <= data[var2]),
        (["var1", "var2"], "x == y", data[var1] == data[var2]),
        (["var1", "var2"], "x != y", data[var1] != data[var2]),
    ]

    for field, test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(Flags({f: flags[f] for f in field}), data[field], func)
        assert (result == expected).all(axis=None)


def test_arithmeticOperators(data):

    var1, *_ = data.columns

    data = data[var1]
    flags = Flags({var1: pd.Series(UNFLAGGED, index=data.index)})

    tests = [
        ("var1 + 100 > 110", data + 100 > 110),
        ("var1 - 100 > 0", data - 100 > 0),
        ("var1 * 100 > 200", data * 100 > 200),
        ("var1 / 100 > .1", data / 100 > 0.1),
        ("var1 % 2 == 1", data % 2 == 1),
        ("var1 ** 2 == 0", data ** 2 == 0),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"processGeneric(func={test})")
        result = _execGeneric(flags, data, func)
        assert (result == expected).all(axis=None)


def test_nonReduncingBuiltins(data):
    var1, *_ = data.columns
    data = data.iloc[1:10, 0]
    flags = Flags({var1: pd.Series(UNFLAGGED, index=data.index)})

    tests = [
        ("abs(x)", np.abs(data)),
        ("log(x)", np.log(data)),
        ("exp(x)", np.exp(data)),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"processGeneric(func={test})")
        result = _execGeneric(flags, data, func)
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
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(Flags({f: flags[f] for f in field}), data[field], func)
        assert (result == expected).all(axis=None)


def test_variableAssignments(data):

    config = f"""
    varname ; test
    dummy1  ; processGeneric(field=["var1", "var2"], func=x + y)
    dummy2  ; flagGeneric(field=["var1", "var2"], func=x + y > 0)
    """

    fobj = writeIO(config)
    saqc = fromConfig(fobj, data)

    expected_columns = set(data.columns) | {"dummy1", "dummy2"}
    assert set(saqc.data.columns) == expected_columns
    assert set(saqc.flags.columns) == expected_columns


def test_processExistingTarget(data):
    config = f"""
    varname ; test
    var2   ; flagMissing()
    var2   ; processGeneric(func=y - 1, flag=DOUBTFUL)
    """

    fobj = writeIO(config)
    saqc = fromConfig(fobj, data)
    assert (saqc._data["var2"] == data["var2"] - 1).all()
    assert len(saqc._flags.history["var2"]) == 2
    assert saqc._flags.history["var2"].hist[0].isna().all()
    assert (saqc._flags.history["var2"].hist[1] == DOUBTFUL).all()


def test_flagTargetExisting(data):
    config = f"""
    varname ; test
    dummy   ; processGeneric(field="var1", func=x < 1)
    dummy   ; processGeneric(field="var2", func=y >1)
    """

    fobj = writeIO(config)
    saqc = fromConfig(fobj, data)
    assert len(saqc.data["dummy"]) == len(saqc.flags["dummy"])


def test_processTargetExistingFail(data_diff):
    config = f"""
    varname ; test
    dummy   ; processGeneric(field="var1", func=x + 1)
    dummy   ; processGeneric(field="var2", func=y - 1)
    """

    fobj = writeIO(config)
    with pytest.raises(ValueError):
        fromConfig(fobj, data_diff)


def test_flagTargetExistingFail(data_diff):
    config = f"""
    varname ; test
    dummy   ; flagGeneric(field="var1", func=x < 1)
    dummy   ; flagGeneric(field="var2", func=y > 1)
    """

    fobj = writeIO(config)
    with pytest.raises(ValueError):
        fromConfig(fobj, data_diff)


def test_callableArgumentsUnary(data):

    window = 5

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def testFuncUnary(data, field, flags, func, **kwargs):
        value = data[field].rolling(window=window).apply(func)
        data[field] = value
        return data, initFlagsLike(data)

    var = data.columns[0]

    config = f"""
    varname ; test
    {var}   ; testFuncUnary(func={{0}})
    """

    tests = [
        # ("sum", np.nansum),
        ("std(exp(x))", lambda x: np.std(np.exp(x))),
    ]

    for (name, func) in tests:
        fobj = writeIO(config.format(name))
        result_config = fromConfig(fobj, data).result.data
        result_api = SaQC(data).testFuncUnary(var, func=func).result.data
        expected = data[var].rolling(window=window).apply(func)
        assert (result_config[var].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var].dropna() == expected.dropna()).all(axis=None)


def test_callableArgumentsBinary(data):
    var1, var2 = data.columns[:2]

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def testFuncBinary(data, field, flags, func, **kwargs):
        data[field] = func(data[var1], data[var2])
        return data, initFlagsLike(data)

    config = f"""
    varname ; test
    {var1}  ; testFuncBinary(func={{0}})
    """

    tests = [
        ("x + y", lambda x, y: x + y),
        ("y - (x * 2)", lambda y, x: y - (x * 2)),
    ]

    for (name, func) in tests:
        fobj = writeIO(config.format(name))
        result_config = fromConfig(fobj, data).result.data
        result_api = SaQC(data).testFuncBinary(var1, func=func).result.data
        expected = func(data[var1], data[var2])
        assert (result_config[var1].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var1].dropna() == expected.dropna()).all(axis=None)


def test_isflagged(data):

    var1, var2, *_ = data.columns
    flags = initFlagsLike(data)
    flags[data[var1].index[::2], var1] = BAD

    tests = [
        ([var1], f"isflagged(x)", flags[var1] > UNFLAGGED),
        ([var1], f"isflagged(x)", flags[var1] >= BAD),
        ([var2], f"~isflagged(x)", flags[var2] == UNFLAGGED),
        (
            [var1, var2],
            f"~(x > 999) & (~isflagged(y))",
            ~(data[var1] > 999) & (flags[var2] == UNFLAGGED),
        ),
    ]

    for field, test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test}, flag=BAD)")
        result = _execGeneric(Flags({f: flags[f] for f in field}), data[field], func)
        assert (result == expected).all(axis=None)
