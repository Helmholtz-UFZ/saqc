#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from typing import List
import astor
import numpy as np
import pandas as pd

from saqc.core.config import Params
from saqc.funcs.register import FUNC_MAP

# Module should be renamed to compiler


def initDslFuncMap(flagger, nodata, level):
    func_map = {
        "abs": {"func": abs, "target": "data"},
        "max": {"func": max, "target": "data"},
        "min": {"func": min, "target": "data"},
        "mean": {"func": np.mean, "target": "data"},
        "sum": {"func": np.sum, "target": "data"},
        "std": {"func": np.std, "target": "data"},
        "len": {"func": len, "target": "data"},
        "isflagged": {
            "func": lambda flags: flagger.isFlagged(flags),
            "target": "flags"},
        "ismissing": {
            "func": lambda data: ((data == nodata) | pd.isnull(data)),
            "target": "data"},
    }
    return {k: v[level] for k, v in func_map.items()}


class DslTransformer(ast.NodeTransformer):
    # TODO: restrict the supported nodes

    SUPPORTED = (
        ast.Expression,
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

    def __init__(self, func_map):
        self.func_map = func_map

    def _rename(self, node: ast.Name, target: str) -> ast.Subscript:

        if node.id == "this":
            slice = ast.Index(value=ast.Name(id="field", ctx=ast.Load()))
        else:
            slice = ast.Index(value=ast.Constant(value=node.id))

        return ast.Subscript(
            value=ast.Name(id=target, ctx=ast.Load()),
            slice=slice,
            ctx=ast.Load())

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.func_map:
            raise TypeError(f"unspported function: {func_name}")

        target = self.func_map[func_name]

        node = ast.Call(
            func=node.func,
            args=[self._rename(node.args[0], target)],
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
            func=ast.Subscript(
                value=ast.Name(id="FUNC_MAP", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=func_name)),
                ctx=ast.Load()),
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
    if not isinstance(tree.body, ast.Call):
        raise TypeError('function call needed')
    return tree


def compileTree(tree: ast.Expression):
    return compile(ast.fix_missing_locations(tree),
                   "<ast>",
                   mode="eval")


def evalCode(code, data, flags, field, flagger, nodata):
    global_env = initDslFuncMap(flagger, nodata, level="func")
    local_env = {
        "FUNC_MAP": FUNC_MAP,
        "data": data, "flags": flags,
        "field": field, "this": field,
        "flagger": flagger, "NODATA": nodata}

    return eval(code, global_env, local_env)


def evalExpression(expr, data, flags, field, flagger, nodata):

    tree = parseExpression(expr)
    dsl_transformer = DslTransformer(initDslFuncMap(flagger, nodata, "target"))
    transformed_tree = MetaTransformer(dsl_transformer).visit(tree)
    code = compileTree(transformed_tree)
    return evalCode(code, data, flags, field, flagger, nodata)
