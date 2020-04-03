#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from test.common import initData, TESTFLAGGER, TESTNODATA
from saqc.core.core import run
from saqc.core.config import Fields as F

from test.common import initData, TESTFLAGGER, TESTNODATA, initMetaDict, initMetaString

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


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_missingIdentifier(data, flagger):

    flagger = flagger.initFlags(data)
    tests = ["flagGeneric(func=fff(var2) < 5)", "flagGeneric(func=var3 != NODATA)"]
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
        _, result_flagger = evalExpression(f"flagGeneric(func={expr})", data, this, flagger, np.nan)
        expected_flagger = flagger.setFlags(this, loc=mask, test="generic")
        assert np.all(result_flagger.isFlagged() == expected_flagger.isFlagged())


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_arithmeticOperators(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = data[var1]

    tests = [
        ("this + 100", this + 100),
        ("this - 1000", this - 1000),
        ("this * 2", this * 2),
        ("this / 100", this / 100),
        ("this % 2", this % 2),
        ("this ** 2", this ** 2),
    ]

    # check within the usually enclosing scope
    for expr, expected in tests:
        result_data, _ = evalExpression(f"procGeneric(func={expr})", data, var1, flagger, np.nan)
        assert np.all(result_data[expected.name] == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_nonReduncingBuiltins(data, flagger):
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = data[var1]

    tests = [
        ("abs(this)", np.abs(this)),
        ("sqrt(this)", np.sqrt(this)),
        ("exp(this)", np.exp(this)),
        ("log(this)", np.log(this)),
    ]

    for expr, expected in tests:
        result_data, _ = evalExpression(f"procGeneric(func={expr})", data, var1, flagger, np.nan)
        assert np.all(result_data[expected.name] == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_reduncingBuiltins(data, flagger, nodata):
    data.loc[::4] = nodata
    flagger = flagger.initFlags(data)
    var1, *_ = data.columns
    this = data[var1]

    tests = [
        ("min(this)", np.min(this)),
        (f"max(this)", np.max(this)),
        (f"sum(this)", np.nansum(this)),
        ("mean(this)", np.nanmean(this)),
        (f"std(this)", np.std(this)),
        (f"len(this)", len(this)),
    ]
    for expr, expected in tests:
        result_data, _ = evalExpression(f"procGeneric(func={expr})", data, var1, flagger, np.nan)
        assert np.all(result_data[var1] == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_ismissing(data, flagger, nodata):

    data.iloc[: len(data) // 2, 0] = np.nan
    data.iloc[(len(data) // 2) + 1 :, 0] = -9999
    var1, *_ = data.columns

    flagger = flagger.initFlags(data)

    tests = [
        (f"ismissing({var1})", lambda data: (pd.isnull(data) | (data == nodata)).all()),
        (f"~ismissing({var1})", lambda data: (pd.notnull(data) & (data != nodata)).all(),),
    ]

    for expr, checkFunc in tests:
        idx = _evalDslExpression(expr, data, var1, flagger, nodata)
        assert checkFunc(data.loc[idx, var1])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("nodata", TESTNODATA)
def test_bitOps(data, flagger, nodata):
    var1, var2, *_ = data.columns
    this = data[var1]

    flagger = flagger.initFlags(data)

    tests = [
        (f"~(this > mean(this))", ~(this > np.nanmean(this))),
        (f"(this <= 0) | (0 < {var1})", (this <= 0) | (0 < data[var1])),
        (f"({var2} >= 0) & (0 > this)", (data[var2] >= 0) & (0 > this)),
    ]

    for expr, expected in tests:
        _, flagger_result = evalExpression(f"flagGeneric(func={expr})", data, this.name, flagger, nodata)
        assert (flagger_result.isFlagged(this.name) == expected).all()


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
        (f"~({var2}>999) & (~isflagged({var2}))", ~(data[var2] > 999) & (~flagger.isFlagged(var2))),
    ]

    for expr, flags_expected in tests:
        _, flagger_result = evalExpression(f"flagGeneric(func={expr})", data, var1, flagger, np.nan)
        flags_result = flagger_result.isFlagged(var1)
        assert np.all(flags_result == flags_expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflaggedArgument(data, flagger):

    var1, var2, *_ = data.columns

    flagger = flagger.initFlags(data).setFlags(var1, iloc=slice(None, None, 2), flag=flagger.BAD)

    tests = [
        (_evalDslExpression(f"isflagged({var1}, BAD)", data, var2, flagger), flagger.isFlagged(var1, flag=flagger.BAD)),
        (
            _evalDslExpression(f"isflagged({var1}, UNFLAGGED, '==')", data, var2, flagger),
            flagger.isFlagged(var1, flag=flagger.UNFLAGGED, comparator="=="),
        ),
    ]

    for result, expected in tests:
        assert np.all(result == expected)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_variableAssignments(data, flagger):
    var1, var2, *_ = data.columns

    from saqc.core.core import run
    from saqc.core.config import Fields as F
    from test.common import writeIO

    config = f"""
    {F.VARNAME}  ; {F.TESTS}
    dummy1       ; procGeneric(func=var1 + var2)
    dummy2       ; flagGeneric(func=var1 + var2 > 0)
    """

    result_data, result_flagger = run(writeIO(config), flagger, data)

    assert set(result_data.columns) == set(data.columns) | {
        "dummy1",
    }
    assert set(result_flagger.getFlags().columns) == set(data.columns) | {"dummy1", "dummy2"}
