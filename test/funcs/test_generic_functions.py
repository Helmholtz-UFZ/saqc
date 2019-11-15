#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from test.common import initData, TESTFLAGGER, TESTNODATA

from saqc.core.evaluator import (
    DslTransformer,
    initDslFuncMap,
    parseExpression,
    evalExpression,
    compileTree,
    evalCode)


def _evalExpression(expr, data, flags, field, flagger, nodata=np.nan):
    dsl_transformer = DslTransformer(initDslFuncMap(nodata), data.columns)
    tree = parseExpression(expr)
    transformed_tree = dsl_transformer.visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, data, flags, field, flagger, nodata)


@pytest.fixture
def data():
    return initData()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagPropagation(data, flagger):
    flags = flagger.setFlags(
        flagger.initFlags(data),
        'var2', iloc=slice(None, None, 5))

    var1, var2, *_ = data.columns
    this = var1
    var2_flags = flagger.isFlagged(flags, var2)
    var2_data = data[var2].mask(var2_flags)
    data, flags = evalExpression(
        "generic(func=var2 < mean(var2))",
        data, flags,
        this,
        flagger, np.nan
    )

    expected = (var2_flags | (var2_data < var2_data.mean()))
    result = flagger.isFlagged(flags, this)
    assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingIdentifier(data, flagger):
    flags = flagger.initFlags(data)
    tests = [
        "generic(func=func(var2) < 5)",
        "generic(func=var3 != NODATA)"
    ]
    for expr in tests:
        with pytest.raises(NameError):
            evalExpression(expr, data, flags, data.columns[0], flagger, np.nan)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_comparisons(data, flagger):
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns
    this = var1

    tests = [
        ("this > 100", data[this] > 100),
        (f"10 >= {var2}", 10 >= data[var2]),
        (f"{var2} < 100", data[var2] < 100),
        (f"this <= {var2}", data[this] <= data[var2])
    ]

    # check within the usually enclosing scope
    for expr, mask in tests:
        _, result_flags = evalExpression(
            f"generic(func={expr})",
            data, flags,
            this, flagger, np.nan)
        expected_flags = flagger.setFlags(flags, this, loc=mask, test="generic")
        assert np.all(
            flagger.isFlagged(result_flags) == flagger.isFlagged(expected_flags)
        )


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_nonReduncingBuiltins(data, flagger):
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns
    this = var1

    tests = [
        ("abs(this)", np.abs(data[this])),
    ]

    for expr, expected in tests:
        result = _evalExpression(expr, data, flags, this, flagger)
        assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_reduncingBuiltins(data, flagger, nodata):
    data.loc[::4] = nodata
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns
    this = var1

    tests = [
        ("min(this)", np.min(data[this])),
        (f"max({var1})", np.max(data[var1])),
        (f"sum({var2})", np.sum(data[var2])),
        ("mean(this)", np.mean(data[this])),
        (f"std({var1})", np.std(data[var1])),
        (f"len({var2})", len(data[var2])),
    ]

    for expr, expected in tests:
        result = _evalExpression(expr, data, flags, this, flagger, nodata)
        assert result == expected



@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_ismissing(data, flagger, nodata):

    data.iloc[:len(data)//2, 0] = np.nan
    data.iloc[(len(data)//2)+1:, 0] = -9999
    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)

    tests = [
        (f"ismissing({var1})", lambda data: (pd.isnull(data) | (data == nodata)).all()),
        (f"~ismissing({var1})", lambda data: (pd.notnull(data) & (data != nodata)).all())
    ]

    for expr, checkFunc in tests:
        idx = _evalExpression(expr, data, flags, "var1", flagger, nodata)
        assert checkFunc(data.loc[idx, var1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_bitOps(data, flagger, nodata):
    var1, var2, *_ = data.columns
    this = var1

    flags = flagger.initFlags(data)

    # TODO: extend the test list
    tests = [
        (f"generic(func=~(this > mean(this)))", ~(data[this] > np.nanmean(data[this]))),
        (f"generic(func=(this <= 0) | (0 < {var1}))", (data[this] <= 0) | (0 < data[var1])),
        (f"generic(func=({var2} >= 0) & (0 > this))", (data[var2] >= 0) | (0 > data[this]))
    ]

    for expr, expected in tests:
        _, flags = evalExpression(expr, data, flags, this, flagger, nodata)
        assert (flagger.isFlagged(flags[this]) == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflagged(data, flagger):

    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2))
    flags = flagger.setFlags(flags, var2, iloc=slice(None, None, 2))

    idx = _evalExpression(f"isflagged({var1})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags, var1)
    assert (flagged == idx).all


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflaggedArgument(data, flagger):

    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2), flag=flagger.BAD)

    idx = _evalExpression(
        f"isflagged({var1}, {flagger.BAD})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags, var1, flagger.BAD, comparator=">=")
    assert (flagged == idx).all()
