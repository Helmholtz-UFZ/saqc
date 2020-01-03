#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from functools import partial
from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.core.config import Params
from saqc.funcs.register import FUNC_MAP


def _dslIsFlagged(flagger, data, flag=None):
    return flagger.isFlagged(data.name, flag=flag)


def initLocalEnv(data: pd.DataFrame, field: str, flagger: BaseFlagger, nodata: float) -> Dict[str, Any]:
    return {
        "data": data,
        "field": field,
        "this": field,
        "flagger": flagger,

        "NAN": np.nan,
        "NODATA": nodata,
        "GOOD": flagger.GOOD,
        "BAD": flagger.BAD,
        "UNFLAGGED": flagger.UNFLAGGED,

        "ismissing": lambda data: ((data == nodata) | pd.isnull(data)),
        "isflagged": partial(_dslIsFlagged, flagger),
        "abs": np.abs,
        "max": np.nanmax,
        "min": np.nanmin,
        "mean": np.nanmean,
        "sum": np.nansum,
        "std": np.nanstd,
        "len": len,
    }


class DslTransformer(ast.NodeTransformer):

    SUPPORTED = (
        ast.Expression,
        ast.UnaryOp, ast.BinOp,
        ast.BitOr, ast.BitAnd,
        ast.Num,
        ast.Compare,
        ast.Add, ast.Sub,
        ast.Mult, ast.Div,
        ast.Pow, ast.Mod,
        ast.USub,
        ast.Eq, ast.NotEq,
        ast.Gt, ast.Lt,
        ast.GtE, ast.LtE,
        ast.Invert,
        ast.Name,
    )

    def __init__(self, environment, variables):
        self.environment = environment
        self.variables = variables


    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.environment:
            raise NameError(f"unspported function: '{func_name}'")

        return ast.Call(
            func=node.func,
            args=[self.visit(arg) for arg in node.args],
            keywords=[],
        )

    def visit_Name(self, node):
        name = node.id
        if name == "this":
            name = self.environment["field"]

        if name in self.variables:
            value = ast.Constant(value=name)
            return ast.Subscript(
                value=ast.Name(id="data", ctx=ast.Load()),
                slice=ast.Index(value=value),
                ctx=ast.Load(),
            )
        if name in self.environment:
            return ast.Constant(value=name)

        raise NameError(f"unknown variable: '{name}'")

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"invalid expression: '{node}'")
        return super().generic_visit(node)


class ConfigTransformer(ast.NodeTransformer):

    SUPPORTED_NODES = (
        ast.Call, ast.Num, ast.Str, ast.keyword,
        ast.NameConstant, ast.UnaryOp, ast.Name,
        ast.Load, ast.Expression, ast.Subscript,
        ast.Index, ast.USub
    )

    SUPPORTED_ARGUMENTS = (
        ast.Str, ast.Num, ast.NameConstant, ast.Call,
        ast.UnaryOp, ast.USub, ast.Name
    )

    def __init__(self, dsl_transformer, environment, pass_parameter):
        self.dsl_transformer = dsl_transformer
        self.environment = environment
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
        if self.func_name == Params.FLAG_GENERIC and key == Params.FUNC:
            node = ast.keyword(arg=key, value=self.dsl_transformer.visit(value))
            return node

        if key not in FUNC_MAP[self.func_name].signature + self.pass_parameter:
            raise TypeError(f"unknown function parameter '{node.arg}'")

        if not isinstance(value, self.SUPPORTED_ARGUMENTS):
            raise TypeError(
                f"invalid argument type '{type(value)}'"
            )

        if isinstance(value, ast.Name) and value.id not in self.environment:
            raise NameError(
                f"unknown variable: {value.id}"
            )

        return self.generic_visit(node)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid node: '{node}'")
        return super().generic_visit(node)


def parseExpression(expr: str) -> ast.AST:
    tree = ast.parse(expr, mode="eval")
    return tree


def compileTree(tree: ast.Expression):
    return compile(ast.fix_missing_locations(tree), "<ast>", mode="eval")


def evalCode(code, global_env=None, local_env=None):
    return eval(code, global_env or {}, local_env or {})


def compileExpression(expr, data, field, flagger, nodata=np.nan):
    local_env = initLocalEnv(data, field, flagger, nodata)
    varmap = set(data.columns.tolist() + flagger.getFlags().columns.tolist())
    tree = parseExpression(expr)
    dsl_transformer = DslTransformer(local_env, varmap)
    transformed_tree = ConfigTransformer(dsl_transformer, local_env, flagger.signature).visit(tree)
    return local_env, compileTree(transformed_tree)


def evalExpression(expr, data, field, flagger, nodata=np.nan):
    # NOTE:
    # mask the already flagged value to make all the functions
    # called on the way through the evaluator ignore flagged values
    mask = flagger.isFlagged()
    data_in = data.mask(mask)
    local_env, code = compileExpression(expr, data_in, field, flagger, nodata)
    data_result, flagger_result = evalCode(code, FUNC_MAP, local_env)
    # NOTE:
    # reinject the original values, as we don't want to loose them
    data_result[mask] = data[mask]
    return data_result, flagger_result
