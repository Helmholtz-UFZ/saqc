#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import ast
import io
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.request import urlopen

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
        except (ValueError, URLError):
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


# Todo: needs a (maybe tiny) docstring!
def fromConfig(fname, *args, **func_kwargs):
    saqc = SaQC(*args, **func_kwargs)
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
            func_name, func_kwargs = ConfigFunctionParser().parse(tree.body)
        except Exception as e:
            raise type(e)(f"failed to parse: {field} ; {expr}") from e

        func_kwargs["field" if "field" not in func_kwargs else "target"] = fld
        try:
            # We explictly route all function calls through SaQC.__getattr__
            # in order to do a FUNC_MAP lookup. Otherwise we wouldn't be able
            # to overwrite exsiting test functions with custom register calls.
            saqc = saqc.__getattr__(func_name)(regex=regex, **func_kwargs)
        except Exception as e:
            raise type(e)(f"failed to execute: {field} ; {expr}") from e

    return saqc
