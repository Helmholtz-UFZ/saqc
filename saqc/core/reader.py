#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re

import numpy as np
import pandas as pd

from saqc.core.config import Fields as F
from saqc.core.evaluator import compileExpression


def _raise(config_row, exc, msg, field=None):
    line_number = config_row[F.LINENUMBER]
    base_msg = f"configuration error in line {line_number}"
    if field:
        base_msg += f", column '{field}'"
    msg = base_msg + ":\n" + msg
    raise exc(msg)


def checkConfig(config_df, data, flagger, nodata):
    for _, config_row in config_df.iterrows():

        var_name = config_row[F.VARNAME]
        if pd.isnull(config_row[F.VARNAME]) or not var_name:
            _raise(
                config_row, SyntaxError, f"non-optional column '{F.VARNAME}' is missing or empty"
            )

        test_fields = config_row.filter(regex=F.TESTS).dropna()
        if test_fields.empty:
            _raise(
                config_row,
                SyntaxError,
                f"at least one test needs to be given for variable",
            )

        for col, expr in test_fields.iteritems():
            if not expr:
                _raise(config_row, SyntaxError, f"field '{col}' may not be empty")
            try:
                compileExpression(expr, data, var_name, flagger, nodata)
            except (TypeError, NameError, SyntaxError) as exc:
                _raise(
                    config_row,
                    type(exc),
                    exc.args[0] + f" (failing statement: '{expr}')",
                    col,
                )
    return config_df


def prepareConfig(config_df, data):
    # ensure column-names are lowercase and have no trailing whitespaces
    config_df.columns = [c.lstrip().lower() for c in config_df.columns]

    # add line numbers and remove comments
    config_df[F.LINENUMBER] = np.arange(len(config_df)) + 2
    try:
        comment_mask = ~config_df.iloc[:, 0].str.startswith("#")
    except AttributeError:
        comment_mask = np.ones(len(config_df), dtype=np.bool)
    config_df = config_df[comment_mask]

    if config_df.empty:
        raise SyntaxWarning("config file is empty or all lines are #commented")

    # NOTE:
    # time slicing support is currently disabled
    # fill missing columns
    # for field in [F.VARNAME, F.START, F.END, F.PLOT]:
    for field in [F.VARNAME, F.PLOT]:
        if field not in config_df:
            config_df = config_df.assign(**{field: np.nan})

    for field in [F.START, F.END]:
        config_df = config_df.assign(**{field: np.nan})

    # fill nans with default values
    config_df = config_df.fillna(
        {
            F.VARNAME: np.nan,
            F.START: data.index.min(),
            F.END: data.index.max(),
            F.PLOT: False,
        }
    )

    # dtype = np.datetime64 if isinstance(data.index, pd.DatetimeIndex) else int
    # config_df[F.START] = config_df[F.START].astype(dtype)
    # config_df[F.END] = config_df[F.END].astype(dtype)

    config_df = _expandVarnameWildcards(config_df, data)

    return config_df


def _expandVarnameWildcards(config_df, data):
    new = []
    for idx, row in config_df.iterrows():
        varname = row[F.VARNAME]
        if varname and not pd.isnull(varname) and varname not in data:
            if varname == "*":
                varname = ".*"
            try:
                variables = data.columns[data.columns.str.match(varname)]
                if variables.empty:
                    variables = [varname]
                for var in variables:
                    row = row.copy()
                    row[F.VARNAME] = var
                    new.append(row)
            except re.error:
                pass
        else:
            new.append(row)
    return pd.DataFrame(new).reset_index(drop=True)


def readConfig(fname):
    return pd.read_csv(fname, delimiter=";", skipinitialspace=True)
