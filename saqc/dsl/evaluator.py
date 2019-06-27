#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import copy
import numbers
import operator as op
from numbers import Number
import numpy as np
import pandas as pd

from ..flagger import BaseFlagger


# supported operators
OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub,
    ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.NotEq: op.ne, ast.Eq: op.eq,
    ast.Gt: op.gt, ast.GtE: op.ge,
    ast.Lt: op.lt, ast.LtE: op.le,
    ast.BitAnd: op.and_, ast.BitOr: op.or_, ast.BitXor: op.xor,
    ast.Invert: op.invert
}


def initFunctionNamespace(nodata, flagger):
    return {
        "abs": (abs, "data"),
        "max": (max, "data"),
        "min": (min, "data"),
        "mean": (np.mean, "data"),
        "sum": (np.sum, "data"),
        "std": (np.std, "data"),
        "len": (len, "data"),
        "ismissing": (lambda d: ((d == nodata) | pd.isnull(d)), "data"),
        "isflagged": (flagger.isFlagged, "flags")
    }


def setKey(d, key, value):
    out = copy.copy(d)
    out[key] = value
    return out


def _raiseNameError(name, expr):
    raise NameError(
        "name '{:}' is not definied (failing expression: '{:}')"
        .format(name, expr))


def evalExpression(expr: str, flagger: BaseFlagger,
                   data: pd.DataFrame, flags: pd.DataFrame,
                   field: str, nodata: Number = np.nan,
                   **namespace: dict) -> np.ndarray:

    # type: (...) -> np.ndarray[bool]

    def _eval(node, namespace):
        # type: (ast.Node, dict) -> None
        # the namespace dictionary should provide the data frame for the device
        # being processed and any additional variables (e.g. NODTA, this)

        if isinstance(node, ast.Num):  # <number>
            return node.n

        elif isinstance(node, ast.UnaryOp):
            return OPERATORS[type(node.op)](
                _eval(node.operand, namespace))

        elif isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](
                _eval(node.left, namespace),
                _eval(node.right, namespace))

        elif isinstance(node, ast.Compare):
            # NOTE: chained comparison not supported yet
            op = OPERATORS[node.ops[0].__class__]
            out = op(_eval(node.left, namespace),
                     _eval(node.comparators[0], namespace))
            return out

        elif isinstance(node, ast.Call):
            # functions out of math are allowed
            # kwargs not supported yet
            try:
                func, target = FUNCTIONS[node.func.id]
            except KeyError:
                _raiseNameError(node.func.id, expr)

            namespace = setKey(namespace, "target", target)
            args = [_eval(n, namespace) for n in node.args]
            return func(*args)

        elif isinstance(node, ast.Name):  # <variable>

            field = namespace.get(node.id, node.id)

            if isinstance(field, numbers.Number):
                # name is not referring to an DataFrame field
                return field

            fidx = namespace["flags"].columns
            if isinstance(fidx, pd.MultiIndex):
                fcols = fidx.get_level_values(0).unique()
            else:
                fcols = fidx.values
            dcols = namespace["data"].columns.values

            try:
                if field in fcols:
                    flagcol = namespace["flags"][field]
                    out = flagcol

                if namespace.get("target") != "flags":

                    if field in dcols:
                        datacol = namespace["data"][field]
                        out = datacol

                    if field in dcols and field in fcols:
                        out = np.ma.masked_array(datacol, mask=flagger.isFlagged(flagcol))

            except KeyError:
                _raiseNameError(field, expr)

            return out

        else:
            raise TypeError(node)

    FUNCTIONS = initFunctionNamespace(nodata, flagger)
    namespace = {**namespace,
                 **{"data": data, "flags": flags, "this": field}}
    return _eval(ast.parse(expr, mode='eval').body, namespace)
    # field = namespace["this"]
    # flags = flag_func(flags=namespace["flags"].loc[to_flag_idx, field])
    # namespace["flags"].loc[to_flag_idx, field] = flags
    # return namespace
