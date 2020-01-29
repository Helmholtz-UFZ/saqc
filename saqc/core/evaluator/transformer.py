#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from saqc.core.config import Params
from typing import Dict, Any


class DslTransformer(ast.NodeTransformer):
    def __init__(self, environment: Dict[str, Any]):
        self.environment = environment
        self.arguments = set()

    def visit_Call(self, node):
        return ast.Call(func=node.func, args=[self.visit(arg) for arg in node.args], keywords=[])

    def visit_Name(self, node):
        name = node.id

        if name == "this":
            name = self.environment["field"]

        if name in self.environment["variables"]:
            value = ast.Constant(value=name)
            node = ast.Subscript(
                value=ast.Name(id="data", ctx=ast.Load()), slice=ast.Index(value=value), ctx=ast.Load(),
            )

        self.arguments.add(name)
        return node


class ConfigTransformer(ast.NodeTransformer):
    def __init__(self, environment):
        self.environment = environment
        self.func_name = None

    def visit_Call(self, node):
        self.func_name = node.func.id

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
                value=ast.List(elts=[ast.Str(s=v) for v in dsl_transformer.arguments], ctx=ast.Load(),),
            )
            return [dsl_func, args]

        return self.generic_visit(node)
