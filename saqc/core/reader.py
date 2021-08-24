#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import numpy as np
import pandas as pd

from saqc.core.core import ColumnSelector
from saqc.core.config import Fields as F
from saqc.core.visitor import ConfigFunctionParser
from saqc.core.lib import ConfigController
from saqc.core.register import FUNC_MAP
from saqc.lib.tools import isQuoted, toSequence


COMMENT = "#"
EMPTY = "None"


def _handleEmptyLines(df):
    if F.VARNAME not in df.columns:
        # at least the first line was empty, so we search the header
        df = df.reset_index()
        i = (df == F.VARNAME).first_valid_index()
        df.columns = df.iloc[i]
        df = df.iloc[i + 1 :]

    # mark empty lines
    mask = (df.isnull() | (df == "")).all(axis=1)
    df.loc[mask] = EMPTY
    return df


def _handleComments(df: pd.DataFrame) -> pd.DataFrame:
    # mark commented lines
    df.loc[df[F.VARNAME].str.startswith(COMMENT)] = EMPTY

    for col in df:
        try:
            df[col] = df[col].str.split(COMMENT, expand=True).iloc[:, 0].str.strip()
        except AttributeError:
            # NOTE:
            # if `df[col]` is not of type string, we know, that
            # there are no comments and the `.str` access fails
            pass

    return df


def _injectOptionalColumns(df):
    # inject optional columns
    if F.PLOT not in df:
        empty = (df == EMPTY).all(axis=1)
        df[F.PLOT] = "False"
        df[empty] = EMPTY
    return df


def _parseConfig(df, translator, data, nodata):
    funcs = []
    for lineno, (_, target, expr) in enumerate(df.itertuples()):
        if target == "None" or pd.isnull(target) or pd.isnull(expr):
            continue

        if isQuoted(target):
            target = target[1:-1]
            target = data.columns[data.columns.str.match(target)]

        tree = ast.parse(expr, mode="eval")
        func_name, kwargs = ConfigFunctionParser().parse(tree.body)
        func = FUNC_MAP[func_name]

        control = ConfigController(lineno=lineno + 2, expression=expr)

        if "flag" in kwargs:
            kwargs["flag"] = translator(kwargs["flag"])

        partial = func.bind(
            **{"nodata": nodata, "to_mask": translator.TO_MASK, **kwargs}
        )

        targets = toSequence(target)

        for target in targets:
            selector = ColumnSelector(
                field=kwargs.get("field", target),
                target=target,
            )
            funcs.append((selector, control, partial))

    return funcs


def readConfig(fname, translator, data, nodata):
    df = pd.read_csv(
        fname,
        sep=r"\s*;\s*",
        engine="python",
        dtype=str,
        quoting=3,
        keep_default_na=False,  # don't replace "" by nan
        skip_blank_lines=False,
    )

    df = _handleEmptyLines(df)
    df = _handleComments(df)

    df[F.VARNAME] = df[F.VARNAME].replace(r"^\s*$", np.nan, regex=True)
    df[F.TEST] = df[F.TEST].replace(r"^\s*$", np.nan, regex=True)
    return _parseConfig(df, translator, data, nodata)
