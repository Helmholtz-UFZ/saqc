#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from functools import partial
from typing import Union

import numpy as np
import pandas as pd

from saqc.core.config import Params
from saqc.funcs.register import FUNC_MAP


class Targets:
    DATA = "data"
    FLAGS = "flags"


def _dslInner(func, data, field, flagger):
    return func(data.mask(flagger.isFlagged(field)))


def _dslIsFlagged(data, field, flagger):
    return flagger.isFlagged(field)


def initGlobalMap():
    out = {
        "abs": partial(_dslInner, np.abs),
        "max": partial(_dslInner, np.nanmax),
        "min": partial(_dslInner, np.nanmin),
        "mean": partial(_dslInner, np.nanmean),
        "sum": partial(_dslInner, np.nansum),
        "std": partial(_dslInner, np.nanstd),
        "len": partial(_dslInner, len),
        "isflagged": _dslIsFlagged,
        "nan": np.nan,
        **FUNC_MAP,
    }
    return out

def initLocalMap(data, field, flagger, nodata):
    return {
        "data": data,
        "field": field,
        "this": field,
        "flagger": flagger,
        "NODATA": nodata,
        "ismissing": lambda data, field, flagger: ((data == nodata) | pd.isnull(data)),
    }


class DslTransformer(ast.NodeTransformer):

    SUPPORTED = (
        ast.Expression,
        ast.UnaryOp,
        ast.BinOp,
        ast.BitOr,
        ast.BitAnd,
        ast.Num,
        ast.Compare,
        ast.Add,
        ast.Mult,
        ast.Pow,
        ast.USub,
        ast.NotEq,
        ast.Gt,
        ast.Lt,
        ast.GtE,
        ast.LtE,
        ast.BitAnd,
        ast.Invert,
    )

    def __init__(self, func_map, variables):
        self.func_map = func_map
        self.variables = variables

    def _rename(
        self, node: ast.Name, target: str
    ) -> Union[ast.Subscript, ast.Name, ast.Constant]:
        name = node.id
        if name == "this":
            value = ast.Name(id="field", ctx=ast.Load())
        else:
            if name not in self.variables:
                raise NameError(f"unknown variable: '{name}'")
            value = ast.Constant(value=name)

        if target == Targets.FLAGS:
            return value
        else:
            out = ast.Subscript(
                value=ast.Name(id=target, ctx=ast.Load()),
                slice=ast.Index(value=value),
                ctx=ast.Load(),
            )
            return out

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.func_map:
            raise NameError(f"unspported function: '{func_name}'")

        node = ast.Call(
            func=node.func,
            args=[
                self._rename(node.args[0], Targets.DATA),
                self._rename(node.args[0], Targets.FLAGS),
                ast.Name(id="flagger", ctx=ast.Load()),
            ],
            keywords=[],
        )
        return node

    def visit_Name(self, node):
        return self._rename(node, "data")

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"invalid expression: '{node}'")
        return super().generic_visit(node)


class MetaTransformer(ast.NodeTransformer):

    SUPPORTED_NODES = (
        ast.Call, ast.Num, ast.Str, ast.keyword,
        ast.NameConstant, ast.UnaryOp, ast.Name,
        ast.Load, ast.Expression, ast.Subscript,
        ast.Index, ast.USub
    )

    SUPPORTED_ARGUMENTS = (
        ast.Str, ast.Num, ast.NameConstant, ast.Call,
        ast.UnaryOp, ast.USub
    )

    def __init__(self, dsl_transformer, pass_parameter):
        self.dsl_transformer = dsl_transformer
        self.pass_parameter = pass_parameter
        self.func_name = None

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in FUNC_MAP:
            raise NameError(f"unknown test function: '{func_name}'")
        if node.args:
            raise TypeError("only keyword arguments are supported")
        self.func_name = func_name

        new_args = [
            ast.Name(id="data", ctx=ast.Load()),
            ast.Name(id="field", ctx=ast.Load()),
            ast.Name(id="flagger", ctx=ast.Load()),
        ]

        node = ast.Call(
            func=node.func, args=new_args + node.args, keywords=node.keywords
        )

        return self.generic_visit(node)

    def visit_keyword(self, node):
        key, value = node.arg, node.value
        if self.func_name == Params.GENERIC and key == Params.FUNC:
            node = ast.keyword(arg=key, value=self.dsl_transformer.visit(value))
            return node

        if key not in FUNC_MAP[self.func_name].signature + self.pass_parameter:
            raise TypeError(f"unknown function parameter '{node.arg}'")

        if not isinstance(value, self.SUPPORTED_ARGUMENTS):
            raise TypeError(
                f"invalid argument type '{type(value)}'"
            )

        return self.generic_visit(node)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid expression: '{node}'")
        return super().generic_visit(node)


def parseExpression(expr: str) -> ast.AST:
    tree = ast.parse(expr, mode="eval")
    # if not isinstance(tree.body, (ast.Call, ast.Compare)):
    #     raise TypeError('function call needed')
    return tree


def compileTree(tree: ast.Expression):
    return compile(ast.fix_missing_locations(tree), "<ast>", mode="eval")


def evalCode(code, global_env, local_env):
    return eval(code, global_env, local_env)


def compileExpression(expr, data, flagger, env):
    varmap = set(data.columns.tolist() + flagger.getFlags().columns.tolist())
    tree = parseExpression(expr)
    dsl_transformer = DslTransformer(env, varmap)
    transformed_tree = MetaTransformer(dsl_transformer, flagger.signature).visit(tree)
    return compileTree(transformed_tree)


def evalExpression(expr, data, field, flagger, nodata=np.nan):

    global_env = initGlobalMap()
    local_env = initLocalMap(data, field, flagger, nodata)
    code = compileExpression(expr, data, flagger, {**global_env, **local_env})
    return evalCode(code, global_env, local_env)
