#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

from saqc.constants import ENVIRONMENT
from saqc.core.register import FUNC_MAP


class ConfigExpressionParser(ast.NodeVisitor):
    """
    Generic configuration functions will be rewritten as lambda functions
    and all defined variables will act as arguments, e.g.:
    ``flagGeneric(func=(x != 4) & (y < 3))`` will be rewritten to
    ``lambda x, y: (x != 4) & (y < 3)``

    The main purpose of this class is to identify all variables used in
    a given generic function and to check that it does not violate the
    restrictions imposed onto generic functions.
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
        # anything not in ENVIRONMENT should be an argument of the generic function
        name = node.id
        if name not in ENVIRONMENT:
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
        ast.Attribute,
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

        if key == "func":
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
        # try:
        exec(co, {**ENVIRONMENT}, self.kwargs)

        # let's do some more validity checks
        if check_tree:
            self.generic_visit(value)

    def generic_visit(self, node):
        if not isinstance(node, self.SUPPORTED_NODES):
            raise TypeError(f"invalid node: '{node}'")
        return super().generic_visit(node)
