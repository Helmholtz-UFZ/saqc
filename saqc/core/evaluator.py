#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from functools import partial

# import astor
import numpy as np
import pandas as pd

from saqc.core.config import Params
from saqc.funcs.register import FUNC_MAP

# Module should be renamed to compiler


def _dslInner(func, data, flags, flagger):
    return func(data.mask(flagger.isFlagged(flags)))


def _dslIsFlagged(data, flags, flagger):
    return flagger.isFlagged(flags)


def initDslFuncMap(nodata):
    return {
        "abs": partial(_dslInner, np.abs),
        "max": partial(_dslInner, np.nanmax),
        "min": partial(_dslInner, np.nanmin),
        "mean": partial(_dslInner, np.nanmean),
        "sum": partial(_dslInner, np.nansum),
        "std": partial(_dslInner, np.nanstd),
        "len": partial(_dslInner, len),
        "isflagged": _dslIsFlagged,
        "ismissing": lambda data, flags, flagger: ((data == nodata) | pd.isnull(data)),
    }


class DslTransformer(ast.NodeTransformer):
    # TODO: restrict the supported nodes

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
        self.variables = set(variables)

    def _rename(self, node: ast.Name, target: str) -> ast.Subscript:

        name = node.id
        if name == "this":
            slice = ast.Index(value=ast.Name(id="field", ctx=ast.Load()))
        else:
            if name not in self.variables:
                raise NameError(f"unknown variable: '{name}'")
            slice = ast.Index(value=ast.Constant(value=name))

        return ast.Subscript(
            value=ast.Name(id=target, ctx=ast.Load()),
            slice=slice,
            ctx=ast.Load())

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.func_map:
            raise NameError(f"unspported function: {func_name}")

        node = ast.Call(
            func=node.func,
            args=[
                self._rename(node.args[0], "data"),
                self._rename(node.args[0], "flags"),
                ast.Name(id="flagger", ctx=ast.Load()),
            ],
            keywords=[]
        )
        return node

    def visit_Name(self, node):
        return self._rename(node, "data")

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"Invalid expression: {node}")
        return super().generic_visit(node)


class MetaTransformer(ast.NodeTransformer):

    SUPPORTED = (
        ast.Call,
        ast.Num,
        ast.Str,
        ast.keyword,
        ast.NameConstant,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Expression,
        ast.Subscript,
        ast.Index)

    def __init__(self, dsl_transformer):
        self.dsl_transformer = dsl_transformer
        self.func_name = None

    def visit_Call(self, node):

        func_name = node.func.id

        if func_name not in FUNC_MAP:
            raise TypeError(f"Unknown test function: {func_name}")

        self.func_name = func_name
        new_args = [ast.Name(id="data", ctx=ast.Load()),
                    ast.Name(id="flags", ctx=ast.Load()),
                    ast.Name(id="field", ctx=ast.Load()),
                    ast.Name(id="flagger", ctx=ast.Load())]

        node = ast.Call(
            func=node.func,
            args=new_args + node.args,
            keywords=node.keywords)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if self.func_name == "generic" and node.arg == Params.FUNC:
            node = ast.keyword(
                arg=node.arg,
                value=self.dsl_transformer.visit(node.value))
            return node
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        # we support all unary ops
        return node

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"Invalid expression: {node}")
        return super().generic_visit(node)


def parseExpression(expr: str) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    # if not isinstance(tree.body, (ast.Call, ast.Compare)):
    #     raise TypeError('function call needed')
    return tree


def compileTree(tree: ast.Expression):
    return compile(ast.fix_missing_locations(tree),
                   "<ast>",
                   mode="eval")


def evalCode(code, data, flags, field, flagger, nodata):
    global_env = initDslFuncMap(nodata)
    local_env = {
        **FUNC_MAP,
        "data": data, "flags": flags,
        "field": field, "this": field,
        "flagger": flagger, "NODATA": nodata}

    return eval(code, global_env, local_env)


def evalExpression(expr, data, flags, field, flagger, nodata=np.nan):

    tree = parseExpression(expr)
    dsl_transformer = DslTransformer(initDslFuncMap(nodata), data.columns)
    transformed_tree = MetaTransformer(dsl_transformer).visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, data, flags, field, flagger, nodata)
