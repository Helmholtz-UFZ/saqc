#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.funcs import register
from saqc.core.evaluator import (
    compileTree,
    parseExpression,
    initLocalEnv,
    ConfigChecker,
    ConfigTransformer,
)

from test.common import TESTFLAGGER, initData


def compileExpression(expr, flagger, nodata=np.nan):
    data = initData()
    field = data.columns[0]
    tree = parseExpression(expr)
    env = initLocalEnv(data, field, flagger.initFlags(data), nodata)
    ConfigChecker(env, flagger.signature).visit(tree)
    transformed_tree = ConfigTransformer(env).visit(tree)
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
        # "func",
        "flagDummy(kwarg=[1, 2, 3])",
        "flagDummy(x=5)",
        "flagDummy(dummy())",
        "flagDummy(kwarg=dummy(this))",
    ]

    for expr in exprs:
        with pytest.raises(TypeError):
            compileExpression(expr, flagger)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_supportedArguments(flagger):
    @register()
    def func(data, field, flagger, kwarg, **kwargs):
        return data, flagger

    exprs = [
        "func(kwarg='str')",
        "func(kwarg=5)",
        "func(kwarg=5.5)",
        "func(kwarg=-5)",
        "func(kwarg=True)",
        "func(kwarg=func())",
    ]
    for expr in exprs:
        compileExpression(expr, flagger)
