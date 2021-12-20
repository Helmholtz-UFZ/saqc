#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import ast
from pathlib import Path
from urllib.request import urlopen
from typing import TextIO

import pandas as pd

from saqc.core.core import SaQC
from saqc.core.visitor import ConfigFunctionParser
from saqc.lib.tools import isQuoted


COMMENT = "#"
SEPARATOR = ";"


def _openFile(fname) -> TextIO:
    if isinstance(fname, (str, Path)):
        try:
            fobj = io.StringIO(urlopen(str(fname)).read().decode("utf-8"))
            fobj.seek(0)
        except ValueError:
            fobj = io.open(fname, "r", encoding="utf-8")
    else:
        fobj = fname

    return fobj


def _closeFile(fobj):
    try:
        fobj.close()
    except AttributeError:
        pass


def readFile(fname) -> pd.DataFrame:

    fobj = _openFile(fname)

    out = []
    for i, line in enumerate(fobj):
        row = line.strip().split(COMMENT, 1)[0]
        if not row:
            # skip over comment line
            continue

        parts = [p.strip() for p in row.split(SEPARATOR)]
        if len(parts) != 2:
            raise RuntimeError(
                "The configuration format expects exactly two columns, one "
                "for the variable name and one for the test to apply, but "
                f"in line {i} we got: \n'{line}'"
            )
        out.append([i + 1] + parts)

    _closeFile(fobj)

    df = pd.DataFrame(
        out[1:],
        columns=[
            "row",
        ]
        + out[0][1:],
    ).set_index("row")
    return df


# Todo: needs (maybe tiny) docstring!
def fromConfig(fname, *args, **kwargs):
    saqc = SaQC(*args, **kwargs)
    config = readFile(fname)

    for _, field, expr in config.itertuples():

        regex = False
        if isQuoted(field):
            fld = field[1:-1]
            regex = True
        else:
            fld = field

        try:
            tree = ast.parse(expr, mode="eval")
            func_name, kwargs = ConfigFunctionParser().parse(tree.body)
        except Exception as e:
            raise type(e)(f"failed to parse: {field} ; {expr}") from e

        kwargs["field" if "field" not in kwargs else "target"] = fld
        try:
            saqc = saqc.__getattr__(func_name)(regex=regex, **kwargs)
        except Exception as e:
            raise type(e)(f"failed to execute: {field} ; {expr}") from e

    return saqc
