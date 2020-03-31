#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import logging

from functools import partial
from typing import Any, Dict

import astor
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.core.config import Params
from saqc.funcs.register import FUNC_MAP
from saqc.core.evaluator.checker import ConfigChecker
from saqc.core.evaluator.transformer import ConfigTransformer


logger = logging.getLogger("SaQC")


def _dslIsFlagged(flagger, field, flag=None, comparator=None):
    if comparator is None:
        return flagger.isFlagged(field, flag=flag)
    return flagger.isFlagged(field, flag=flag, comparator=comparator)


def initLocalEnv(data: pd.DataFrame, field: str, flagger: BaseFlagger, nodata: float) -> Dict[str, Any]:

    return {
        # general
        "data": data,
        "field": field,
        "flagger": flagger,
        "this": field,
        # transformation only
        "variables": set(flagger.getFlags().columns.tolist()),
        "nolookup": set(["isflagged"]),  # no variable lookup for flagger based functions,
        # missing values/data
        "NAN": np.nan,
        "NODATA": nodata,
        # flags
        "GOOD": flagger.GOOD,
        "BAD": flagger.BAD,
        "UNFLAGGED": flagger.UNFLAGGED,
        # special functions
        "ismissing": lambda data: ((data == nodata) | pd.isnull(data)),
        "isflagged": partial(_dslIsFlagged, flagger),
        # math
        "abs": np.abs,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "max": np.nanmax,
        "min": np.nanmin,
        "mean": np.nanmean,
        "sum": np.nansum,
        "std": np.nanstd,
        "len": lambda data: np.array(len(data)),
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
    tree = parseExpression(expr)
    ConfigChecker(local_env, flagger.signature).visit(tree)
    transformed_tree = ConfigTransformer(local_env).visit(tree)
    src = astor.to_source(transformed_tree).strip()
    logger.debug(f"calling transformed function:\n{src}")
    return local_env, compileTree(transformed_tree)


def evalExpression(expr, data, field, flagger, nodata=np.nan):
    # mask the already flagged value to make all the functions
    # called on the way through the evaluator ignore flagged values
    mask = flagger.isFlagged()
    data_in = data.copy()
    data_in[mask] = np.nan
    local_env, code = compileExpression(expr, data_in, field, flagger, nodata)
    data_result, flagger_result = evalCode(code, FUNC_MAP, local_env)
    # reinject the original values, as we don't want to loose them
    data_result[mask] = data[mask]
    return data_result, flagger_result
