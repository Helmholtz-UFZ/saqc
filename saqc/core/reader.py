#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re
import logging
from csv import reader
from typing import Dict, List, Any, Union, Iterable, Iterator, Tuple
from contextlib import contextmanager
from io import StringIO, TextIOWrapper

import pandas as pd

from saqc.core.config import Fields as F
from saqc.core.evaluator import compileExpression
from saqc.flagger import BaseFlagger


logger = logging.getLogger("SaQC")


# typing declarations
Config = Iterable[Dict[str, Any]]
Filename = Union[StringIO, str]


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
def _open(fname: Filename) -> Union[StringIO, TextIOWrapper]:
    if isinstance(fname, StringIO):
        yield fname
    else:
        f = open(fname)
        yield f
        f.close()


def _matchKey(keys: Iterable[str], fuzzy_key: str) -> str:
    for key in keys:
        if re.match(fuzzy_key, key):
            return key


def _castRow(row: Dict[str, Any]):
    out = {}
    for row_key, row_value in row.items():
        for fuzzy_key, func in CONFIG_TYPES.items():
            if re.match(fuzzy_key, row_key):
                try:
                    out[row_key] = func(row_value)
                except ValueError:
                    _raise(row, ValueError, f"invalid value: '{row_value}'")
    return out


def _expandVarnameWildcards(config: Config, data: pd.DataFrame) -> Config:
    def isQuoted(string):
        return bool(re.search(r"'.*'|\".*\"", string))

    new = []
    for row in config:
        varname = row[F.VARNAME]
        if varname and isQuoted(varname):
            pattern = varname[1:-1]
            expansion = data.columns[data.columns.str.match(pattern)]
            if not len(expansion):
                logger.warning(f"no match for regular expression '{pattern}'")
            for var in expansion:
                new.append({**row, F.VARNAME: var})
        else:
            new.append(row)
    return new


def _clearRows(rows: Iterable[List[str]], comment: str = "#") -> Iterator[Tuple[str, List[Any]]]:
    for i, row in enumerate(rows):
        row = [c.strip() for c in row]
        if any(row) and not row[0].lstrip().startswith(comment):
            row = [c.split(comment)[0].strip() for c in row]
            yield i, row


def readConfig(fname: Filename, data: pd.DataFrame, sep: str = ";", comment: str = "#") -> pd.DataFrame:
    defaults = {F.VARNAME: "", F.START: data.index.min(), F.END: data.index.max(), F.PLOT: False}

    with _open(fname) as f:
        rdr = reader(f, delimiter=";")

        rows = _clearRows(rdr)
        _, header = next(rows)

        config = []
        for n, row in rows:
            row = dict(zip(header, row))
            row = _castRow({**defaults, **row, F.LINENUMBER: n + 1})
            config.append(row)

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
