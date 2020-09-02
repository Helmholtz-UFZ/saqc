#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast

import numpy as np

import pandas as pd

from saqc.core.config import Fields as F
from saqc.core.visitor import ConfigFunctionParser

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


def _handleComments(df):
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


def _parseConfig(df):
    to_call = []
    for lineno, (_, field, expr, plot) in enumerate(df.itertuples()):
        if field == "None":
            continue
        if pd.isnull(field):
            raise SyntaxError(f"line {lineno}: non-optional column '{F.VARNAME}' missing")
        if pd.isnull(expr):
            raise SyntaxError(f"line {lineno}: non-optional column '{F.TEST}' missing")
        tree = ast.parse(expr, mode="eval")
        cp = ConfigFunctionParser(tree.body)
        to_call.append((cp.func, field, cp.kwargs, plot, lineno + 2, expr))
    return to_call


def readConfig(fname):
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
    df = _injectOptionalColumns(df)
    df = _handleComments(df)

    df[F.VARNAME] = df[F.VARNAME].replace(r"^\s*$", np.nan, regex=True)
    df[F.TEST] = df[F.TEST].replace(r"^\s*$", np.nan, regex=True)
    df[F.PLOT] = df[F.PLOT].replace({"False": "", EMPTY: "", np.nan: ""})
    df = df.astype({F.PLOT: bool})
    df = _parseConfig(df)

    return df
