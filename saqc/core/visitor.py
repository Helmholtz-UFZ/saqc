#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

import numpy as np
import pandas as pd

from saqc.common import *
from saqc.core.register import FUNC_MAP
import saqc.lib.ts_operators as ts_ops


ENVIRONMENT = {
    "NAN": np.nan,
    "abs": np.abs,
    "max": np.nanmax,
    "min": np.nanmin,
    "mean": np.nanmean,
    "sum": np.nansum,
    "std": np.nanstd,
    "len": len,
    "exp": np.exp,
    "log": np.log,
    "var": np.nanvar,
    "median": np.nanmedian,
    "first": ts_ops.first,
    "last": ts_ops.last,
    "count": ts_ops.count,
    "deltaT": ts_ops.deltaT,
    "id": ts_ops.identity,
    "diff": ts_ops.difference,
    "relDiff": ts_ops.relativeDifference,
    "deriv": ts_ops.derivative,
    "rateOfChange": ts_ops.rateOfChange,
    "scale": ts_ops.scale,
    "normScale": ts_ops.normScale,
    "meanStandardize": ts_ops.standardizeByMean,
    "medianStandardize": ts_ops.standardizeByMedian,
    "zLog": ts_ops.zeroLog,
}

RESERVED = {"GOOD", "BAD", "UNFLAGGED", "NODATA"}


class ConfigExpressionParser(ast.NodeVisitor):
    """
    Generic configuration functions will be rewritten as lambda functions
    and variables that need a look up in `data` will act as arguments, e.g.:
      `flagGeneric(func=(x != NODATA) & (y < 3))`
      will be rewritten to
      `lambda x, y: (x != NODATA) & (y < 3)`

    The main purpose of this class is to identify all such lambda arguments
    and check the given expression for accordance with the restrictions
    imposed onto generic functions.
    """

    SUPPORTED = (
        ast.Str,
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

    def __init__(self, node):
        self._args = []
        self.visit(node)
        if not self._args:
            # NOTE:
            # we assume, that we are not dealing with an
            # expression as we couldn't find any arguments
            raise TypeError("not a valid expression")

    @property
    def args(self):
        return tuple(dict.fromkeys(self._args))

    def visit_Call(self, node):
        # only non-keyword arguments allowed
        # in generic functions
        for n in node.args:
            self.visit(n)

    def visit_Name(self, node):
        # NOTE:
        # the assumption is, that anything not in
        # ENVIRONMENT + RESERVED needs a lookup in `data`
        name = node.id
        if name not in ENVIRONMENT and name not in RESERVED:
            self._args.append(name)
        self.generic_visit(node)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(f"invalid expression: '{node}'")
        return super().generic_visit(node)


class ConfigFunctionParser(ast.NodeVisitor):

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
        ast.List,
        ast.Attribute
    )

    def __init__(self, flagger):

        self.kwargs = {}
        self.environment = {
            "GOOD": GOOD,
            "BAD": BAD,
            "UNFLAGGED": UNFLAGGED,
            **ENVIRONMENT,
        }

    def parse(self, node):
        func = self.visit_Call(node)
        return func, self.kwargs

    def visit_Call(self, node):
        if not isinstance(node, ast.Call):
            raise TypeError("expected function call")

        if node.args:
            raise TypeError("only keyword arguments are supported")

        if isinstance(node.func, ast.Attribute):
            func_name = f"{node.func.value.id}.{node.func.attr}"  # type: ignore
        else:
            func_name = node.func.id  # type: ignore

        if func_name not in FUNC_MAP:
            raise NameError(f"unknown function '{func_name}'")

        self.generic_visit(node)
        return func_name

    def visit_keyword(self, node):

        k, v = node.arg, node.value
        check_tree = True

        # NOTE: `node` is not a constant or a variable,
        #       so it should be a function call
        try:
            visitor = ConfigExpressionParser(v)
            args = ast.arguments(
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                args=[ast.arg(arg=a, annotation=None) for a in visitor.args],
                kwarg=None,
                vararg=None,
            )
            v = ast.Lambda(args=args, body=v)
            # NOTE:
            # don't pass the generated functions down
            # to the checks implemented in this class...
            check_tree = False
        except TypeError:
            pass

        vnode = ast.Assign(targets=[ast.Name(id=k, ctx=ast.Store())], value=v)

        # NOTE:
        # in order to get concrete values out of the AST
        # we compile and evaluate the keyword (e.g. max=100)
        # into the dictionary `self.kwargs`
        # -> after all keywords where visited we end up with
        #    a dictionary holding all the passed arguments as
        #    real python objects
        co = compile(
            ast.fix_missing_locations(ast.Interactive(body=[vnode])),
            "<ast>",
            mode="single"
        )
        # NOTE: only pass a copy to not clutter the self.environment
        exec(co, {**self.environment}, self.kwargs)

        # let's do some more validity checks
        if check_tree:
            self.generic_visit(v)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid node: '{node}'")
        return super().generic_visit(node)
