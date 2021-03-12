#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import pytest
import numpy as np
import pandas as pd
import dios

from saqc.common import *
from saqc.flagger import Flagger, initFlagsLike
from saqc.core.visitor import ConfigFunctionParser
from saqc.core.config import Fields as F
from saqc.core.register import register
from saqc.funcs.generic import _execGeneric
from saqc import SaQC

from test.common import TESTNODATA, initData, writeIO


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
    return dios.DictOfSeries(data={col0.name: col0.iloc[: mid + offset], col1.name: col1.iloc[mid - offset :],})


def _compileGeneric(expr, flagger):
    tree = ast.parse(expr, mode="eval")
    _, kwargs = ConfigFunctionParser(flagger).parse(tree.body)
    return kwargs["func"]


def test_missingIdentifier(data):
    flagger = Flagger()

    # NOTE:
    # - the error is only raised at runtime during parsing would be better
    tests = [
        "fff(var2) < 5",
        "var3 != NODATA",
    ]

    for test in tests:
        func = _compileGeneric(f"generic.flag(func={test})", flagger)
        with pytest.raises(NameError):
            _execGeneric(flagger, data, func, field="", nodata=np.nan)


def test_syntaxError():
    flagger = Flagger()
    tests = [
        "range(x=5",
        "rangex=5)",
        "range[x=5]" "range{x=5}" "int->float(x=4)" "int*float(x=4)",
    ]

    for test in tests:
        with pytest.raises(SyntaxError):
            _compileGeneric(f"flag(func={test})", flagger)


def test_typeError():
    """
    test that forbidden constructs actually throw an error
    TODO: find a few more cases or get rid of the test
    """
    flagger = Flagger()

    # : think about cases that should be forbidden
    tests = ("lambda x: x * 2",)

    for test in tests:
        with pytest.raises(TypeError):
            _compileGeneric(f"generic.flag(func={test})", flagger)


def test_comparisonOperators(data):
    flagger = initFlagsLike(data)
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
        func = _compileGeneric(f"generic.flag(func={test})", flagger)
        result = _execGeneric(flagger, data, func, field=var1, nodata=np.nan)
        assert np.all(result == expected)


def test_arithmeticOperators(data):
    flagger = initFlagsLike(data)
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
        func = _compileGeneric(f"generic.process(func={test})", flagger)
        result = _execGeneric(flagger, data, func, field=var1, nodata=np.nan)
        assert np.all(result == expected)


def test_nonReduncingBuiltins(data):
    flagger = initFlagsLike(data)
    var1, *_ = data.columns
    this = var1
    mean = data[var1].mean()

    tests = [
        (f"abs({this})", np.abs(data[this])),
        (f"log({this})", np.log(data[this])),
        (f"exp({this})", np.exp(data[this])),
        (f"ismissing(mask({this} < {mean}))", data[this].mask(data[this] < mean).isna()),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"generic.process(func={test})", flagger)
        result = _execGeneric(flagger, data, func, field=this, nodata=np.nan)
        assert (result == expected).all()


@pytest.mark.parametrize("nodata", TESTNODATA)
def test_reduncingBuiltins(data, nodata):
    data.loc[::4] = nodata
    flagger = initFlagsLike(data)
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
        func = _compileGeneric(f"generic.process(func={test})", flagger)
        result = _execGeneric(flagger, data, func, field=this.name, nodata=nodata)
        assert result == expected


@pytest.mark.parametrize("nodata", TESTNODATA)
def test_ismissing(data, nodata):

    flagger = initFlagsLike(data)
    data.iloc[: len(data) // 2, 0] = np.nan
    data.iloc[(len(data) // 2) + 1 :, 0] = -9999
    this = data.iloc[:, 0]

    tests = [
        (f"ismissing({this.name})", (pd.isnull(this) | (this == nodata))),
        (f"~ismissing({this.name})", (pd.notnull(this) & (this != nodata))),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"generic.flag(func={test})", flagger)
        result = _execGeneric(flagger, data, func, this.name, nodata)
        assert np.all(result == expected)


@pytest.mark.parametrize("nodata", TESTNODATA)
def test_bitOps(data, nodata):
    var1, var2, *_ = data.columns
    this = var1

    flagger = initFlagsLike(data)

    tests = [
        ("~(this > mean(this))", ~(data[this] > np.nanmean(data[this]))),
        (f"(this <= 0) | (0 < {var1})", (data[this] <= 0) | (0 < data[var1])),
        (f"({var2} >= 0) & (0 > this)", (data[var2] >= 0) & (0 > data[this])),
    ]

    for test, expected in tests:
        func = _compileGeneric(f"generic.flag(func={test})", flagger)
        result = _execGeneric(flagger, data, func, this, nodata)
        assert np.all(result == expected)


def test_isflagged(data):

    var1, var2, *_ = data.columns
    flagger = initFlagsLike(data)
    flagger[data[var1].index[::2], var1] = BAD

    tests = [
        (f"isflagged({var1})", flagger[var1] > UNFLAGGED),
        (f"isflagged({var1}, flag=BAD)", flagger[var1] >= BAD),
        (f"isflagged({var1}, UNFLAGGED, '==')", flagger[var1] == UNFLAGGED),
        (f"~isflagged({var2})", flagger[var2] == UNFLAGGED),
        (f"~({var2}>999) & (~isflagged({var2}))", ~(data[var2] > 999) & (flagger[var2] == UNFLAGGED)),
    ]

    for i, (test, expected) in enumerate(tests):
        try:
            func = _compileGeneric(f"generic.flag(func={test}, flag=BAD)", flagger)
            result = _execGeneric(flagger, data, func, field=None, nodata=np.nan)
            assert np.all(result == expected)
        except Exception:
            print(i, test)
            raise

    # test bad combination
    for comp in ['>', '>=', '==', '!=', '<', '<=']:
        fails = f"isflagged({var1}, comparator='{comp}')"

        func = _compileGeneric(f"generic.flag(func={fails}, flag=BAD)", flagger)
        with pytest.raises(ValueError):
            _execGeneric(flagger, data, func, field=None, nodata=np.nan)


def test_variableAssignments(data):
    var1, var2, *_ = data.columns

    config = f"""
    {F.VARNAME}  ; {F.TEST}
    dummy1       ; generic.process(func=var1 + var2)
    dummy2       ; generic.flag(func=var1 + var2 > 0)
    """

    fobj = writeIO(config)
    saqc = SaQC(data).readConfig(fobj)
    result_data, result_flagger = saqc.getResult(raw=True)

    assert set(result_data.columns) == set(data.columns) | {
        "dummy1",
    }
    assert set(result_flagger.columns) == set(data.columns) | {"dummy1", "dummy2"}


# TODO: why this must(!) fail ? - a comment would be helpful
@pytest.mark.xfail(strict=True)
def test_processMultiple(data_diff):
    var1, var2, *_ = data_diff.columns

    config = f"""
    {F.VARNAME} ; {F.TEST}
    dummy       ; generic.process(func=var1 + 1)
    dummy       ; generic.process(func=var2 - 1)
    """

    fobj = writeIO(config)
    saqc = SaQC(data_diff).readConfig(fobj)
    result_data, result_flagger = saqc.getResult()
    assert len(result_data["dummy"]) == len(result_flagger["dummy"])


def test_callableArgumentsUnary(data):

    window = 5

    @register(masking='field')
    def testFuncUnary(data, field, flagger, func, **kwargs):
        data[field] = data[field].rolling(window=window).apply(func)
        return data, initFlagsLike(data)

    var = data.columns[0]

    config = f"""
    {F.VARNAME} ; {F.TEST}
    {var}       ; testFuncUnary(func={{0}})
    """

    tests = [
        ("sum", np.nansum),
        ("std(exp(x))", lambda x: np.std(np.exp(x))),
    ]

    for (name, func) in tests:
        fobj = writeIO(config.format(name))
        result_config, _ = SaQC(data).readConfig(fobj).getResult()
        result_api, _ = SaQC(data).testFuncUnary(var, func=func).getResult()
        expected = data[var].rolling(window=window).apply(func)
        assert (result_config[var].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var].dropna() == expected.dropna()).all(axis=None)


def test_callableArgumentsBinary(data):
    var1, var2 = data.columns[:2]

    @register(masking='field')
    def testFuncBinary(data, field, flagger, func, **kwargs):
        data[field] = func(data[var1], data[var2])
        return data, initFlagsLike(data)

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
        result_config, _ = SaQC(data).readConfig(fobj).getResult()
        result_api, _ = SaQC(data).testFuncBinary(var1, func=func).getResult()
        expected = func(data[var1], data[var2])
        assert (result_config[var1].dropna() == expected.dropna()).all(axis=None)
        assert (result_api[var1].dropna() == expected.dropna()).all(axis=None)
