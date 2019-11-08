#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from ..common import initData, TESTFLAGGER

from saqc.dsl.parser import (
    DslTransformer,
    initDslFuncMap,
    parseExpression,
    compileTree,
    evalCode)


def _evalExpression(expr, data, flags, field, flagger, nodata=np.nan):
    dsl_transformer = DslTransformer(initDslFuncMap(flagger, nodata, "target"))
    tree = parseExpression(expr)
    transformed_tree = dsl_transformer.visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, data, flags, "var1", flagger, nodata)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_ismissing(flagger):

    nodata = -9999

    data = initData()
    data.iloc[:len(data)//2, 0] = np.nan
    data.iloc[(len(data)//2)+1:, 0] = nodata
    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)

    idx = _evalExpression(f"ismissing({var1})", data, flags, "var1", flagger, nodata)
    fdata = data.loc[idx, var1]
    assert (pd.isnull(fdata) | (fdata == nodata)).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflagged(flagger):

    data = initData()
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2))
    flags = flagger.setFlags(flags, var2, iloc=slice(None, None, 2))

    idx = _evalExpression(f"isflagged({var1})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags[var1])
    assert (flagged == idx).all


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isflaggedArgument(flagger):

    data = initData()
    var1, var2, *_ = data.columns

    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, var1, iloc=slice(None, None, 2), flag=flagger.BAD)

    idx = _evalExpression(
        f"isflagged({var1}, {flagger.BAD})", data, flags, var2, flagger)

    flagged = flagger.isFlagged(flags[var1], flagger.BAD, comparator=">=")
    assert (flagged == idx).all()
