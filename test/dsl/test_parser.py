#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from saqc.dsl.parser import (
    compileTree,
    parseExpression,
    initDslFuncMap,
    DslTransformer,
    MetaTransformer)

from saqc.test.common import TESTFLAGGER


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


# def test_parsingGeneric():
#     test_expr = "ismissing(this)"
#     expr = f"generic(func={test_expr})"
#     expected_nodes = [type(n) for n in ast.walk(ast.parse(test_expr, mode="eval").body)]

#     expected_name, expected_kwargs = ("generic", {"func": expected_nodes})
#     result_name, result_kwargs = parseFlag(expr)
#     result_nodes = [type(n) for n in ast.walk(result_kwargs[Params.FUNC])]

#     assert result_name == expected_name
#     assert result_kwargs.keys() == expected_kwargs.keys()
#     assert result_nodes == expected_nodes

# @pytest.fixture #(scope="module")
# def data():
#     return initData()


# TESTFLAGGERS = [
#     SimpleFlagger(),
# ]


# @pytest.mark.parametrize("flagger", TESTFLAGGERS)
# def test_parsingBasic(data, flagger):

#     exprs = [
#         "range(min=5, max=4)",
#         # "generic(func=ismissing(this))"
#     ]

#     for expr in exprs:
#         tree = compileExpression(expr)
#         code = compile(tree, "<ast>", mode="eval")
#         eval(code,
#              {"FUNC_MAP": FUNC_MAP},
#              {"data": data,
#               "flags": flagger.initFlags(data),
#               "flagger": flagger,
#               "field": data.columns[0]})
