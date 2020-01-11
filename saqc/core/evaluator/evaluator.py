#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from functools import partial
from typing import Any, Dict

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
        "variables": set(data.columns.tolist() + flagger.getFlags().columns.tolist()),
    }


class DslChecker(ast.NodeVisitor):

    SUPPORTED = (
        ast.Expression,
        ast.UnaryOp,
        ast.BinOp,
        ast.BitOr,
        ast.BitAnd,
        ast.Num,
        ast.Compare,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.Lt,
        ast.GtE,
        ast.LtE,
        ast.Invert,
        ast.Name,
        ast.Load,
        ast.Call,
    )

    def __init__(self, environment):
        self.environment = environment

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.environment:
            raise NameError(f"unspported function: '{func_name}'")
        self.generic_visit(node)

    def visit_Name(self, node):
        name = node.id
        if name != "this" and name not in self.environment and name not in self.environment["variables"]:
            raise NameError(f"unknown variable: '{name}'")
        self.generic_visit(node)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"invalid expression: '{node}'")
        return super().generic_visit(node)


class ConfigChecker(ast.NodeVisitor):

    SUPPORTED_NODES = (
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
        ast.Index,
        ast.USub,
    )

    SUPPORTED_ARGUMENTS = (ast.Str, ast.Num, ast.NameConstant, ast.Call, ast.UnaryOp, ast.USub, ast.Name)

    def __init__(self, environment, pass_parameter):
        self.pass_parameter = pass_parameter
        self.environment = environment
        self.func_name = None

    def visit_Call(self, node):

        func_name = node.func.id
        if func_name not in FUNC_MAP:
            raise NameError(f"unknown test function: '{func_name}'")
        if node.args:
            raise TypeError("only keyword arguments are supported")
        self.func_name = func_name
        return self.generic_visit(node)

    def visit_keyword(self, node):
        key, value = node.arg, node.value
        if self.func_name == Params.FLAG_GENERIC and key == Params.FUNC:
            DslChecker(self.environment).visit(value)
            return

        if key not in FUNC_MAP[self.func_name].signature + self.pass_parameter:
            raise TypeError(f"unknown function parameter '{node.arg}'")

        if not isinstance(value, self.SUPPORTED_ARGUMENTS):
            raise TypeError(f"invalid argument type '{type(value)}'")

        if isinstance(value, ast.Name) and value.id not in self.environment:
            raise NameError(f"unknown variable: {value.id}")

        return self.generic_visit(node)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid node: '{node}'")
        return super().generic_visit(node)


class DslTransformer(ast.NodeTransformer):
    def __init__(self, environment: Dict[str, Any]):
        self.environment = environment
        self.arguments = set()
        self.invert = False
        self.func_name = None

    def visit_Invert(self, node):
        self.invert = True
        return node

    def visit_Call(self, node):
        self.func_name = node.func.id
        return ast.Call(func=node.func, args=[self.visit(arg) for arg in node.args], keywords=[])

    def visit_Name(self, node):
        name = node.id
        if name == "this":
            name = self.environment["field"]

        # NOTE:
        # we need a way to prevent some variables
        # from ending up in `flagGeneric`, see the
        # problem with np.all(~isflagged(x)) is True
        if self.func_name == "isflagged" and self.invert:
            self.invert = False
        else:
            self.arguments.add(name)

        if name in self.environment["variables"]:
            value = ast.Constant(value=name)
            node = ast.Subscript(
                value=ast.Name(id="data", ctx=ast.Load()), slice=ast.Index(value=value), ctx=ast.Load(),
            )
        elif name in self.environment:
            node = ast.Constant(value=name)

        return node


class ConfigTransformer(ast.NodeTransformer):
    def __init__(self, environment):
        self.environment = environment
        self.func_name = None

    def visit_Call(self, node):
        func_name = node.func.id
        self.func_name = func_name

        new_args = [
            ast.Name(id="data", ctx=ast.Load()),
            ast.Name(id="field", ctx=ast.Load()),
            ast.Name(id="flagger", ctx=ast.Load()),
        ]

        node = ast.Call(func=node.func, args=new_args + node.args, keywords=node.keywords)

        return self.generic_visit(node)

    def visit_keyword(self, node):
        key, value = node.arg, node.value

        if self.func_name == Params.FLAG_GENERIC and key == Params.FUNC:
            dsl_transformer = DslTransformer(self.environment)
            value = dsl_transformer.visit(value)
            dsl_func = ast.keyword(arg=key, value=value)
            # NOTE:
            # Inject the additional `func_arguments` argument `flagGeneric`
            # expects, to keep track of all the touched variables. We
            # need this to propagate the flags from the independent variables
            args = ast.keyword(
                arg=Params.GENERIC_ARGS,
                value=ast.List(elts=[ast.Str(s=v) for v in dsl_transformer.arguments], ctx=ast.Load()),
            )
            return [dsl_func, args]

        return self.generic_visit(node)


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
    ConfigChecker(local_env, flagger.signature).visit(tree)
    transformed_tree = ConfigTransformer(local_env).visit(tree)
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
