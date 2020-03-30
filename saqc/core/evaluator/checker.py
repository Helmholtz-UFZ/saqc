#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

from saqc.funcs.register import FUNC_MAP
from saqc.core.config import Params


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
        if name not in self.environment and name not in self.environment["variables"]:
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

    SUPPORTED_ARGUMENTS = (
        ast.Str,
        ast.Num,
        ast.NameConstant,
        ast.Call,
        ast.UnaryOp,
        ast.USub,
        ast.Name,
    )

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
        if self.func_name in (Params.FLAG_GENERIC, Params.PROC_GENERIC) and key == Params.FUNC:
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
