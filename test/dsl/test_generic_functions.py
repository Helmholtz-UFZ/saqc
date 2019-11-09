#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

import pytest
import numpy as np
import pandas as pd

from ..common import initData, TESTFLAGGER

from saqc.dsl.parser import (
    DslTransformer,
    initDslFuncMap,
    evalExpression,
    compileTree,
    evalCode)


def _evalExpression(expr, data, flags, field, flagger, nodata=np.nan):
    dsl_transformer = DslTransformer(initDslFuncMap(nodata))
    tree = ast.parse(expr, mode="eval")
    transformed_tree = dsl_transformer.visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, data, flags, "var1", flagger, nodata)


@pytest.fixture
def data():
    return initData()


@pytest.fixture
def nodata():
    return -99990


# @pytest.mark.parametrize("flagger", TESTFLAGGER)
# def test_flagPropagation(data, flagger):
#     flags = flagger.setFlags(
#         flagger.initFlags(data),
#         'var2', iloc=slice(None, None, 5))

#     var1, var2, *_ = data.columns
#     this = var1
#     var2_flags = flagger.isFlagged(flags[var2])
#     var2_data = data[var2].mask(var2_flags)
#     data, flags = evalExpression(
#         "generic(func=var2 < mean(var2))",
#         data, flags,
#         this,
#         flagger, np.nan
#     )

#     expected = (var2_flags | (var2_data < var2_data.mean()))
#     result = flagger.isFlagged(flags[this])
#     assert (result == expected).all()


# @pytest.mark.parametrize("flagger", TESTFLAGGER)
# def test_missingIdentifier(data, flagger):
#     flags = flagger.initFlags(data)
#     tests = [
#         "func(var2) < 5",
#         "var3 != NODATA"
#     ]
#     for expr in tests:
#         with pytest.raises(NameError):
#             _evalExpression(expr, data, flags, data.columns[0], flagger)


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

    # check directly
    for expr, expected in tests:
        result = _evalExpression(expr, data, flags, this, flagger, np.nan)
        assert (result == expected).all()

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
        result = _evalExpression(expr, data, flags, this, flagger, np.nan)
        assert (result == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_reduncingBuiltins(data, flagger):
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
        result = _evalExpression(expr, data, flags, this, flagger, np.nan)
        assert result == expected



@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_ismissing(data, flagger, nodata):

    data.iloc[:len(data)//2, 0] = np.nan
    data.iloc[(len(data)//2)+1:, 0] = nodata
    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)

    idx = _evalExpression(f"ismissing({var1})", data, flags, "var1", flagger, nodata)
    fdata = data.loc[idx, var1]
    assert (pd.isnull(fdata) | (fdata == nodata)).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflagged(data, flagger):

    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2))
    flags = flagger.setFlags(flags, var2, iloc=slice(None, None, 2))

    idx = _evalExpression(f"isflagged({var1})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags[var1])
    assert (flagged == idx).all


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflaggedArgument(data, flagger):

    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2), flag=flagger.BAD)

    idx = _evalExpression(
        f"isflagged({var1}, {flagger.BAD})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags[var1], flagger.BAD, comparator=">=")
    assert (flagged == idx).all()
