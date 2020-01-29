#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from test.common import initData, TESTFLAGGER, TESTNODATA
from saqc.core.core import run
from saqc.core.config import Fields as F

from test.common import initData, TESTFLAGGER, TESTNODATA, initMetaDict

from saqc.core.evaluator import (
    DslTransformer,
    initLocalEnv,
    parseExpression,
    evalExpression,
    compileTree,
    evalCode,
)


def _evalDslExpression(expr, data, field, flagger, nodata=np.nan):
    env = initLocalEnv(data, field, flagger, nodata)
    tree = parseExpression(expr)
    transformed_tree = DslTransformer(env).visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, local_env=env)


@pytest.fixture
def data():
    return initData()


# @pytest.mark.parametrize("flagger", TESTFLAGGER)
# def test_flagPropagation(data, flagger):
#     var1, var2, *_ = data.columns
#     this = var1

#     flagger = flagger.initFlags(data).setFlags(var2, iloc=slice(None, None, 5))

#     var2_flags = flagger.isFlagged(var2)
#     var2_data = data[var2].mask(var2_flags)
#     data, flagger_result = evalExpression(
#         "flagGeneric(func=var2 < mean(var2))", data, this, flagger, np.nan
#     )

#     expected = var2_flags | (var2_data < var2_data.mean())
#     result = flagger_result.isFlagged(this)
#     assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingIdentifier(data, flagger):

    flagger = flagger.initFlags(data)
    tests = [
        "flagGeneric(func=fff(var2) < 5)",
        "flagGeneric(func=var3 != NODATA)"
    ]
    for expr in tests:
        with pytest.raises(NameError):
            evalExpression(expr, data, data.columns[0], flagger, np.nan)


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

    # check within the usually enclosing scope
    for expr, mask in tests:
        _, result_flagger = evalExpression(
            f"flagGeneric(func={expr})", data, this, flagger, np.nan
        )
        expected_flagger = flagger.setFlags(this, loc=mask, test="generic")
        assert np.all(result_flagger.isFlagged() == expected_flagger.isFlagged())


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_arithmeticOperators(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = data[var1]

    tests = [
        ("this + 100 > 110", this + 100 > 110),
        ("this - 100 > 0", this - 100 > 0),
        ("this * 100 > 200", this * 100 > 200),
        ("this / 100 > .1", this / 100 > .1),
        ("this % 2 == 1", this % 2 == 1),
        ("this ** 2 == 0", this ** 2 == 0),
    ]

    # check within the usually enclosing scope
    for expr, mask in tests:
        _, result_flagger = evalExpression(
            f"flagGeneric(func={expr})", data, var1, flagger, np.nan
        )
        expected_flagger = flagger.setFlags(var1, loc=mask, test="generic")
        assert np.all(result_flagger.isFlagged() == expected_flagger.isFlagged())


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_nonReduncingBuiltins(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = var1

    tests = [
        ("abs(this)", np.abs(data[this])),
    ]

    for expr, expected in tests:
        result = _evalDslExpression(expr, data, this, flagger)
        assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_reduncingBuiltins(data, flagger, nodata):
    data.loc[::4] = nodata
    flagger = flagger.initFlags(data)
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
        result = _evalDslExpression(expr, data, this, flagger, nodata)
        assert result == expected


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_ismissing(data, flagger, nodata):

    data.iloc[: len(data) // 2, 0] = np.nan
    data.iloc[(len(data) // 2) + 1 :, 0] = -9999
    var1, *_ = data.columns

    flagger = flagger.initFlags(data)

    tests = [
        (f"ismissing({var1})", lambda data: (pd.isnull(data) | (data == nodata)).all()),
        (
            f"~ismissing({var1})",
            lambda data: (pd.notnull(data) & (data != nodata)).all(),
        ),
    ]

    for expr, checkFunc in tests:
        idx = _evalDslExpression(expr, data, var1, flagger, nodata)
        assert checkFunc(data.loc[idx, var1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_bitOps(data, flagger, nodata):
    var1, var2, *_ = data.columns
    this = var1

    flagger = flagger.initFlags(data)

    tests = [
        (f"flagGeneric(func=~(this > mean(this)))", ~(data[this] > np.nanmean(data[this]))),
        (
            f"flagGeneric(func=(this <= 0) | (0 < {var1}))",
            (data[this] <= 0) | (0 < data[var1]),
        ),
        (
            f"flagGeneric(func=({var2} >= 0) & (0 > this))",
            (data[var2] >= 0) & (0 > data[this]),
        ),
    ]

    for expr, expected in tests:
        _, flagger_result = evalExpression(expr, data, this, flagger, nodata)
        assert (flagger_result.isFlagged(this) == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflagged(data, flagger):

    flagger = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flagger = flagger.setFlags(var1, iloc=slice(None, None, 2))
    flagger = flagger.setFlags(var2, iloc=slice(None, None, 2))

    idx = _evalDslExpression(f"isflagged({var1})", data, var2, flagger)

    flagged = flagger.isFlagged(var1)
    assert (flagged == idx).all


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_invertIsFlagged(data, flagger):

    flagger = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flagger = flagger.setFlags(var2, iloc=slice(None, None, 2))

    tests = [
        (f"~isflagged({var2})", ~flagger.isFlagged(var2)),
        (f"~({var2}>999) & (~isflagged({var2}))", ~(data[var2] > 999) & (~flagger.isFlagged(var2)))
    ]

    for expr, flags_expected in tests:
        _, flagger_result = evalExpression(
            f"flagGeneric(func={expr})", data, var1, flagger, np.nan
        )
        flags_result = flagger_result.isFlagged(var1)
        assert np.all(flags_result == flags_expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflaggedArgument(data, flagger):

    var1, var2, *_ = data.columns

    flagger = flagger.initFlags(data).setFlags(
        var1, iloc=slice(None, None, 2), flag=flagger.BAD
    )

    tests = [
        (_evalDslExpression(f"isflagged({var1}, BAD)", data, var2, flagger),
         flagger.isFlagged(var1, flag=flagger.BAD)
        ),
        (_evalDslExpression(f"isflagged({var1}, UNFLAGGED, '==')", data, var2, flagger),
         flagger.isFlagged(var1, flag=flagger.UNFLAGGED, comparator="==")),
    ]

    for result, expected in tests:
        assert np.all(result == expected)

