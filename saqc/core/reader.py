#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from .config import Fields as F
from .evaluator import compileExpression


def _raise(config_row, exc, msg, field=None):
    line_number = config_row[F.LINENUMBER]
    base_msg = f"configuration error in line {line_number}"
    if field:
        base_msg += f", column '{field}'"
    msg = base_msg + ":\n" + msg
    raise exc(msg)


def checkConfig(config_df, data, flags, flagger, nodata):
    for _, config_row in config_df.iterrows():
        if pd.isnull(config_row[F.VARNAME]):
            # NOTE: better messages needed
            _raise(config_row, SyntaxError,
                   f"non-optional column '{F.VARNAME}' is missing")

        test_fields = config_row.filter(regex=F.TESTS)
        if test_fields.isna().all():
            _raise(config_row,  SyntaxError,
                   f"at least one test needs to be given vor variable")

        var_name = config_row[F.VARNAME]
        if var_name not in data.columns and not config_row[F.ASSIGN]:
            _raise(config_row, NameError,
                   f"unknown variable '{var_name}'")

        for col, expr in test_fields.iteritems():
            try:
                compileExpression(expr, data, flags, flagger, nodata)
            except (TypeError, NameError, SyntaxError) as exc:
                _raise(config_row, type(exc),
                       exc.args[0] + f" (failing statement: '{expr}')",
                       col)


def prepareConfig(config_df, data):
    # add line numbers and remove comments
    config_df[F.LINENUMBER] = np.arange(len(config_df)) + 1
    try:
        comment_mask = ~config_df.iloc[:, 0].str.startswith("#")
    except AttributeError:
        comment_mask = np.ones(len(config_df), dtype=np.bool)
    config_df = config_df[comment_mask]

    # no dates given, fall back to the available index range
    for field in [F.VARNAME, F.TESTS, F.START, F.END, F.ASSIGN, F.PLOT]:
        if field not in config_df:
            config_df = config_df.assign(**{field: np.nan})

    # fill with default values
    config_df = config_df.fillna({
        F.VARNAME: np.nan,
        F.TESTS: np.nan,
        F.START: data.index.min(),
        F.END: data.index.max(),
        F.ASSIGN: False,
        F.PLOT: False,
    })

    dtype = np.datetime64 if isinstance(data.index, pd.DatetimeIndex) else int

    config_df[F.START] = config_df[F.START].astype(dtype)
    config_df[F.END] = config_df[F.END].astype(dtype)

    return config_df


def readConfig(fname):
    return pd.read_csv(fname, delimiter=",")
