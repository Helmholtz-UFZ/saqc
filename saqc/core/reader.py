#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re

from typing import Dict, List, Any, Union
from contextlib import contextmanager
from io import StringIO, TextIOWrapper

import numpy as np
import pandas as pd

from saqc.core.config import Fields as F
from saqc.core.evaluator import compileExpression
from saqc.flagger import BaseFlagger


ConfigList = List[Dict[str, Any]]


CONFIG_TYPES = {
    F.VARNAME: str,
    F.START: pd.to_datetime,
    F.END: pd.to_datetime,
    F.TESTS: str,
    F.PLOT: lambda v: str(v).lower() == "true",
    F.LINENUMBER: int,
}


def _raise(config_row, exc, msg, field=None):
    line_number = config_row[F.LINENUMBER]
    base_msg = f"configuration error in line {line_number}"
    if field:
        base_msg += f", column '{field}'"
    msg = base_msg + ":\n" + msg
    raise exc(msg)


@contextmanager
def _open(fname: str) -> Union[StringIO, TextIOWrapper]:
    if isinstance(fname, StringIO):
        yield fname
    else:
        f = open(fname)
        yield f
        f.close()


def _parseRow(row: str, sep: str, comment: str) -> List[str]:
    """
    remove in column comments, mainly needed to allow end line comments
    """
    return [c.split(comment)[0].strip() for c in row.split(sep)]


def _castRow(row: Dict[str, str]) -> Dict[str, Any]:
    """
    cast values to the data type given in 'types'
    """
    out = {}
    keys = pd.Index(row.keys())
    for k, func in CONFIG_TYPES.items():
        try:
            key = keys[keys.str.match(k)][0]
        except IndexError:
            continue
        value = row[key]
        # NOTE:
        # this check and the raise should be moved to checkConfig
        try:
            out[key] = func(value)
        except ValueError:
            _raise(row, ValueError, f"invalid value: '{value}'")
    return out


def _expandVarnameWildcards(config: ConfigList, data: pd.DataFrame) -> ConfigList:
    new = []
    for row in config:
        varname = row[F.VARNAME]
        if varname and varname not in data:
            expansion = data.columns[data.columns.str.match(varname)]
            if not len(expansion):
                expansion = [varname]
            for var in expansion:
                new.append({**row, F.VARNAME: var})
        else:
            new.append(row)
    return new


def readConfig(fname: str, data: pd.DataFrame, sep: str = ";", comment: str = "#") -> pd.DataFrame:

    defaults = {
        F.VARNAME: "",
        F.START: data.index.min(),
        F.END: data.index.max(),
        F.PLOT: False,
    }

    with _open(fname) as f:
        content = f.readlines()

    header: List = None
    config: ConfigList = []
    for i, line in enumerate(content):
        line = line.strip()
        if line.startswith(comment) or not line:
            continue
        row = _parseRow(line, sep, comment)
        if header is None:
            header = row
            continue
        values = dict(zip(header, row))
        values = {**defaults, **values, F.LINENUMBER: i + 1}
        config.append(_castRow(values))

    expanded = _expandVarnameWildcards(config, data)
    return pd.DataFrame(expanded)


def checkConfig(config_df: pd.DataFrame, data: pd.DataFrame, flagger: BaseFlagger, nodata: float) -> pd.DataFrame:
    for _, config_row in config_df.iterrows():

        var_name = config_row[F.VARNAME]
        if pd.isnull(config_row[F.VARNAME]) or not var_name:
            _raise(
                config_row, SyntaxError, f"non-optional column '{F.VARNAME}' is missing or empty",
            )

        test_fields = config_row.filter(regex=F.TESTS).dropna()
        if test_fields.empty:
            _raise(
                config_row, SyntaxError, f"at least one test needs to be given for variable",
            )

        for col, expr in test_fields.iteritems():
            if not expr:
                _raise(config_row, SyntaxError, f"field '{col}' may not be empty")
            try:
                compileExpression(expr, data, var_name, flagger, nodata)
            except (TypeError, NameError, SyntaxError) as exc:
                _raise(
                    config_row, type(exc), exc.args[0] + f" (failing statement: '{expr}')", col,
                )
    return config_df
