#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

import pytest
import numpy as np
import pandas as pd

from dios import DictOfSeries

from test.common import TESTFLAGGER, TESTNODATA, initData, writeIO
from saqc.core.visitor import ConfigFunctionParser
from saqc.core.config import Fields as F
from saqc.core.register import register
from saqc import SaQC, SimpleFlagger
from saqc.funcs.functions import _execGeneric


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
    return DictOfSeries(data={col0.name: col0.iloc[: mid + offset], col1.name: col1.iloc[mid - offset :],})


def _compileGeneric(expr):
    tree = ast.parse(expr, mode="eval")
    cp = ConfigFunctionParser(tree.body)
    return cp.kwargs["func"]


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingIdentifier(data, flagger):

    # NOTE:
    # - the error is only raised at runtime during parsing would be better
    tests = [
        "fff(var2) < 5",
        "var3 != NODATA",
    ]

    for test in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        with pytest.raises(NameError):
            _execGeneric(flagger, data, func, field="", nodata=np.nan)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_syntaxError(flagger):

    tests = [
        "range(x=5",
        "rangex=5)",
        "range[x=5]" "range{x=5}" "int->float(x=4)" "int*float(x=4)",
    ]

    for test in tests:
        with pytest.raises(SyntaxError):
            _compileGeneric(f"flagGeneric(func={test})")


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_typeError(flagger):

    """
    test that forbidden constructs actually throw an error
    TODO: find a few more cases or get rid of the test
    """

    # : think about cases that should be forbidden
    tests = ("lambda x: x * 2",)

    for test in tests:
        with pytest.raises(TypeError):
            _compileGeneric(f"flagGeneric(func={test})")


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_comparisonOperators(data, flagger):
    flagger = flagger.initFlags(data)
    var1, var2, *_ = data.columns
    this = var1

    tests = [
        ("this > 100", data[this] > 100),
        (f"10 >= {var2}", 10 >= data[var2]),
        (f"{var2} < 100", data[var2] < 100),
        (f"this <= {var2}", data[this] <= data[var2]),
        (f"{var1} == {var2}", data[this] == data[var2]),
        (f"{var1} != {var2}", data[this] != data[var2]),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(flagger, data, func, field=var1, nodata=np.nan)
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_arithmeticOperators(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = data[var1]

    tests = [
        ("var1 + 100 > 110", this + 100 > 110),
        ("var1 - 100 > 0", this - 100 > 0),
        ("var1 * 100 > 200", this * 100 > 200),
        ("var1 / 100 > .1", this / 100 > 0.1),
        ("var1 % 2 == 1", this % 2 == 1),
        ("var1 ** 2 == 0", this ** 2 == 0),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"procGeneric(func={test})")
        result = _execGeneric(flagger, data, func, field=var1, nodata=np.nan)
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_nonReduncingBuiltins(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = var1

    tests = [
        (f"abs({this})", np.abs(data[this])),
        (f"log({this})", np.log(data[this])),
        (f"exp({this})", np.exp(data[this])),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"procGeneric(func={test})")
        result = _execGeneric(flagger, data, func, field=this, nodata=np.nan)
        assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_reduncingBuiltins(data, flagger, nodata):

    data.loc[::4] = nodata
    flagger = flagger.initFlags(data)
    var1 = data.columns[0]
    this = data.iloc[:, 0]

    tests = [
        ("min(this)", np.nanmin(this)),
        (f"max({var1})", np.nanmax(this)),
        (f"sum({var1})", np.nansum(this)),
        ("mean(this)", np.nanmean(this)),
        (f"std({this.name})", np.std(this)),
        (f"len({this.name})", len(this)),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"procGeneric(func={test})")
        result = _execGeneric(flagger, data, func, field=this.name, nodata=nodata)
        assert result == expected


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_ismissing(data, flagger, nodata):

    data.iloc[: len(data) // 2, 0] = np.nan
    data.iloc[(len(data) // 2) + 1 :, 0] = -9999
    this = data.iloc[:, 0]

    tests = [
        (f"ismissing({this.name})", (pd.isnull(this) | (this == nodata))),
        (f"~ismissing({this.name})", (pd.notnull(this) & (this != nodata))),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(flagger, data, func, this.name, nodata)
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_bitOps(data, flagger, nodata):
    var1, var2, *_ = data.columns
    this = var1

    flagger = flagger.initFlags(data)

    tests = [
        ("~(this > mean(this))", ~(data[this] > np.nanmean(data[this]))),
        (f"(this <= 0) | (0 < {var1})", (data[this] <= 0) | (0 < data[var1])),
        (f"({var2} >= 0) & (0 > this)", (data[var2] >= 0) & (0 > data[this])),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(flagger, data, func, this, nodata)
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflagged(data, flagger):

    var1, var2, *_ = data.columns

    flagger = flagger.initFlags(data).setFlags(var1, loc=data[var1].index[::2], flag=flagger.BAD)

    tests = [
        (f"isflagged({var1})", flagger.isFlagged(var1)),
        (f"isflagged({var1}, BAD)", flagger.isFlagged(var1, flag=flagger.BAD, comparator=">=")),
        (f"isflagged({var1}, UNFLAGGED, '==')", flagger.isFlagged(var1, flag=flagger.UNFLAGGED, comparator="==")),
        (f"~isflagged({var2})", ~flagger.isFlagged(var2)),
        (f"~({var2}>999) & (~isflagged({var2}))", ~(data[var2] > 999) & (~flagger.isFlagged(var2))),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"flagGeneric(func={test})")
        result = _execGeneric(flagger, data, func, field=None, nodata=np.nan)
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_variableAssignments(data, flagger):
    var1, var2, *_ = data.columns

    config = f"""
    {F.VARNAME}  ; {F.TEST}
    dummy1       ; procGeneric(func=var1 + var2)
    dummy2       ; flagGeneric(func=var1 + var2 > 0)
    """

    fobj = writeIO(config)
    saqc = SaQC(flagger, data).readConfig(fobj)
    result_data, result_flagger = saqc.getResult()

    assert set(result_data.columns) == set(data.columns) | {
        "dummy1",
    }
    assert set(result_flagger.getFlags().columns) == set(data.columns) | {"dummy1", "dummy2"}


@pytest.mark.xfail(stric=True)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_procGenericMultiple(data_diff, flagger):
    var1, var2, *_ = data_diff.columns

    config = f"""
    {F.VARNAME} ; {F.TEST}
    dummy       ; procGeneric(func=var1 + 1)
    dummy       ; procGeneric(func=var2 - 1)
    """

    fobj = writeIO(config)
    saqc = SaQC(flagger, data_diff).readConfig(fobj)
    result_data, result_flagger = saqc.getResult()
    assert len(result_data["dummy"]) == len(result_flagger.getFlags("dummy"))


def test_callableArgumentsUnary(data):

    window = 5

    @register(masking='field')
    def testFuncUnary(data, field, flagger, func, **kwargs):
        data[field] = data[field].rolling(window=window).apply(func)
        return data, flagger.initFlags(data=data)

    flagger = SimpleFlagger()
    var = data.columns[0]

    config = f"""
    {F.VARNAME} ; {F.TEST}
    {var}      ; testFuncUnary(func={{0}})
    """

    tests = [
        ("sum", np.sum),
        ("std(exp(x))", lambda x: np.std(np.exp(x))),
    ]

    for (name, func) in tests:
        fobj = writeIO(config.format(name))
        result_config, _ = SaQC(flagger, data).readConfig(fobj).getResult()
        result_api, _ = SaQC(flagger, data).testFuncUnary(var, func=func).getResult()
        expected = data[var].rolling(window=window).apply(func)
        assert (result_config[var].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var].dropna() == expected.dropna()).all(axis=None)


def test_callableArgumentsBinary(data):

    flagger = SimpleFlagger()
    var1, var2 = data.columns[:2]

    @register(masking='field')
    def testFuncBinary(data, field, flagger, func, **kwargs):
        data[field] = func(data[var1], data[var2])
        return data, flagger.initFlags(data=data)

    config = f"""
    {F.VARNAME} ; {F.TEST}
    {var1}      ; testFuncBinary(func={{0}})
    """

    tests = [
        ("x + y", lambda x, y: x + y),
        ("y - (x * 2)", lambda y, x: y - (x * 2)),
    ]

    for (name, func) in tests:
        fobj = writeIO(config.format(name))
        result_config, _ = SaQC(flagger, data).readConfig(fobj).getResult()
        result_api, _ = SaQC(flagger, data).testFuncBinary(var1, func=func).getResult()
        expected = func(data[var1], data[var2])
        assert (result_config[var1].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var1].dropna() == expected.dropna()).all(axis=None)
