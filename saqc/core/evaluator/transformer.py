#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

from typing import Dict, Any
from contextlib import contextmanager

from saqc.core.config import Params


class DslTransformer(ast.NodeTransformer):
    def __init__(self, environment: Dict[str, Any]):
        self.environment = environment

    def visit_Call(self, node):
        new_args = node.args
        for a in new_args:
            a.lookup = node.func.id not in self.environment["nolookup"]

        node = ast.Call(func=node.func, args=new_args, keywords=[])
        return self.generic_visit(node)

    def visit_Name(self, node):

        # NOTE:
        #
        # There are different categories of name nodes:
        #
        # 1. Names that need a lookup in the global/local eval
        #    environment (e.g. function names, dsl constants, ...)
        #    -> nodes need to leave visit_Name unaltered
        # 2. Names that need a lookup in the 'data' DataFrame
        #    -> nodes need to be rewritten int ast.Subscript
        # 3. Names that should be treated as constants and be passed to
        #    functions requiring a 'field' parameter (e.g. 'isflagged')
        #    -> nodes need to be rewritten to ast.Constant/ast.Str
        #
        # TODO:
        #
        # The differentiation between these categories is done based
        # on the two variables out of 'self.environment', namely
        # 'nolookup' and 'variables' in two different methods
        # ('vsisit_Call' and 'visit_Name'). This continues to feel hacky
        # and I really like to see a cleaner solution for that problem

        name = node.id

        if name == "this":
            name = self.environment["this"]

        if name in self.environment["variables"]:
            # determine further tree-transformation path by target
            if getattr(node, "lookup", True):
                value = ast.Constant(value=name)
                node = ast.Subscript(
                    value=ast.Name(id="data", ctx=ast.Load()), slice=ast.Index(value=value), ctx=ast.Load(),
                )
            else:
                node = ast.Constant(value=name)

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

        if self.func_name in (Params.FLAG_GENERIC, Params.PROC_GENERIC) and key == Params.FUNC:
            dsl_transformer = DslTransformer(self.environment)
            value = dsl_transformer.visit(value)
            return ast.keyword(arg=key, value=value)

        return self.generic_visit(node)
