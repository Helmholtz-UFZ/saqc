#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import ast
import importlib

from saqc.core.register import FUNC_MAP
from saqc.options import parsing
from saqc.parsing.environ import ENVIRONMENT


class ConfigExpressionParser(ast.NodeVisitor):
    SUPPORTED = (
        ast.Expression,
        ast.Constant,
        ast.UnaryOp,
        ast.BinOp,
        ast.BitOr,
        ast.BitAnd,
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
        ast.Attribute,
        # ast.Lambda,
    )

    def __init__(self, node):
        self._args = []
        self.visit(node)

    @property
    def args(self):
        return tuple(dict.fromkeys(self._args))

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED):
            raise TypeError(
                f"invalid generic function construct: {type(node).__name__}"
            )
        return super().generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise TypeError("only variable loading is allowed")

        name = node.id
        if name not in ENVIRONMENT:
            self._args.append(name)

        self.generic_visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, (ast.Attribute, ast.Name)):
            raise TypeError("invalid call target")

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name.startswith("_"):
                raise TypeError("private methods are not allowed")

            if method_name not in parsing.allowed_series_methods:
                raise AttributeError(f"unknown attribute: '{method_name}'")

            # Visit the object the method is called on:
            self.visit(node.func.value)

        for arg in node.args:
            self.visit(arg)

        for kw in node.keywords:
            if kw.arg is None:
                raise TypeError("**kwargs are not allowed")
            self.visit(kw.value)

    # def visit_Lambda(self, node):
    #     pass

    def visit_Attribute(self, node):
        # Attribute access is only allowed as a method call handled in visit_Call.
        # This rejects ds1.index, ds1.values, ds1.__class__, etc.
        raise TypeError("attribute access is only allowed for whitelisted method calls")


class ConfigFunctionParser(ast.NodeVisitor):
    SUPPORTED_NODES = (
        ast.Call,
        ast.Constant,
        ast.keyword,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Expression,
        ast.Subscript,
        ast.Index,
        ast.USub,
        ast.Attribute,
        ast.List,
        ast.Tuple,
        ast.Dict,
    )

    def __init__(self):
        self.kwargs = {}

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
        key, value = node.arg, node.value
        check_tree = True

        imports = {}
        if key == "func":
            if (isinstance(value, ast.Name) and value.id in ENVIRONMENT) or (
                isinstance(value, ast.Constant) and value.value in ENVIRONMENT
            ):
                func = ENVIRONMENT[
                    value.id if isinstance(value, ast.Name) else value.value
                ]
                # handle the missing attribute for numpy.ufunc
                module = getattr(func, "__module__", "numpy")
                if module.startswith("saqc"):
                    # if it's an saqc function, we need to import the top level package first
                    imports["saqc"] = importlib.import_module("saqc")
                imports[module] = importlib.import_module(module)
                value = ast.parse(f"{module}.{func.__name__}").body[0].value
            else:
                visitor = ConfigExpressionParser(value)
                args = ast.arguments(
                    posonlyargs=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    args=[ast.arg(arg=a, annotation=None) for a in visitor.args],
                    kwarg=None,
                    vararg=None,
                )
                value = ast.Lambda(args=args, body=value)
            # NOTE:
            # don't pass the generated functions down
            # to the checks implemented in this class...
            check_tree = False

        vnode = ast.Assign(targets=[ast.Name(id=key, ctx=ast.Store())], value=value)

        # NOTE:
        # in order to get concrete values out of the AST
        # we compile and evaluate every keyword (e.g. max=100)
        # into the dictionary `self.kwargs`
        # -> after all keywords where visited we end up with
        # a dictionary holding all the passed arguments as
        # real python objects
        co = compile(
            ast.fix_missing_locations(ast.Interactive(body=[vnode])),
            "<ast>",
            mode="single",
        )
        # NOTE: only pass a copy to not clutter the ENVIRONMENT
        exec(co, {**ENVIRONMENT, **imports}, self.kwargs)

        # let's do some more validity checks
        if check_tree:
            self.generic_visit(value)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid node: '{node}'")
        return super().generic_visit(node)
