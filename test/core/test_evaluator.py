#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.funcs import register
from saqc.core.evaluator import (
    compileTree,
    parseExpression,
    initLocalEnv,
    DslTransformer,
    ConfigTransformer,
)

from test.common import TESTFLAGGER, dummyRegisterFunc, initData


def compileExpression(expr, flagger, nodata=np.nan):
    data = initData()
    field = data.columns[0]
    tree = parseExpression(expr)
    env = initLocalEnv(data, field, flagger.initFlags(data), nodata)
    dsl_transformer = DslTransformer(env, {})
    transformed_tree = ConfigTransformer(dsl_transformer, env, flagger.signature).visit(tree)
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

    register("func")(dummyRegisterFunc)
    register("otherFunc")(dummyRegisterFunc)

    exprs = [
        # "func",
        "func(kwarg=[1, 2, 3])",
        "func(x=5)",
        "func(otherFunc())",
        "func(kwarg=otherFunc(this))",
    ]

    for expr in exprs:
        with pytest.raises(TypeError):
            compileExpression(expr, flagger)




@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_supportedArguments(flagger):


    register("func")(dummyRegisterFunc)
    register("otherFunc")(dummyRegisterFunc)

    exprs = [
        "func(kwarg='str')",
        "func(kwarg=5)",
        "func(kwarg=5.5)",
        "func(kwarg=-5)",
        "func(kwarg=True)",
        "func(kwarg=otherFunc())",
    ]
    for expr in exprs:
        compileExpression(expr, flagger)


