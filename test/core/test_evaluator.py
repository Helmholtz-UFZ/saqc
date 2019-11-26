#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.core.evaluator import (
    compileTree,
    parseExpression,
    initDslFuncMap,
    DslTransformer,
    MetaTransformer)

from test.common import TESTFLAGGER


def compileExpression(expr, flagger, nodata=np.nan):
    tree = parseExpression(expr)
    dsl_transformer = DslTransformer(initDslFuncMap(flagger, nodata, "target"))
    transformed_tree = MetaTransformer(dsl_transformer).visit(tree)
    code = compileTree(transformed_tree)
    return code


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_syntaxError(flagger):
    exprs = [
        "range(x=5",
        "rangex=5)",
        "range[x=5]" "range{x=5}" "int->float(x=4)" "int*float(x=4)",
    ]

    for expr in exprs:
        with pytest.raises(SyntaxError):
            compileExpression(expr, flagger)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_typeError(flagger):
    exprs = [
        "range",
        "nodata(x=[1, 2, 3])",
        "nodata(func=ismissing(this))",
        "range(deleteEverything())",
    ]

    for expr in exprs:
        with pytest.raises(TypeError):
            compileExpression(expr, flagger)
