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
from saqc.core.evaluator.checker import ConfigChecker
from saqc.core.evaluator.transformer import ConfigTransformer


def _dslIsFlagged(flagger, data, flag=None):
    return flagger.isFlagged(data.name, flag=flag)

def _dslAggregateFunc(func, flagger, data):
    flags = flagger.isFlagged(data.name)
    return func(data.mask(flags))

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
        "mean": partial(_dslAggregateFunc, np.nanmean, flagger),
        "sum": partial(_dslAggregateFunc, np.nansum, flagger),
        "std": partial(_dslAggregateFunc, np.nanstd, flagger),
        "len": len,
        "variables": set(data.columns.tolist() + flagger.getFlags().columns.tolist()),
    }


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
    # mask = flagger.isFlagged()
    # data_in = data.mask(mask)
    local_env, code = compileExpression(expr, data, field, flagger, nodata)
    data_result, flagger_result = evalCode(code, FUNC_MAP, local_env)
    # NOTE:
    # reinject the original values, as we don't want to loose them
    # data_result[mask] = data[mask]
    return data_result, flagger_result
