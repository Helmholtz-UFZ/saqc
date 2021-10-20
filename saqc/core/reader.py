#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import ast
from pathlib import Path
import pandas as pd

from saqc.core.core import SaQC
from saqc.core.visitor import ConfigFunctionParser
from saqc.lib.tools import isQuoted


COMMENT = "#"
SEPARATOR = ";"


def readFile(fname):

    fobj = (
        io.open(fname, "r", encoding="utf-8")
        if isinstance(fname, (str, Path))
        else fname
    )

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
        out.append(
            [
                i + 1,
            ]
            + parts
        )

    if isinstance(fname, str):
        fobj.close()

    df = pd.DataFrame(
        out[1:],
        columns=[
            "row",
        ]
        + out[0][1:],
    ).set_index("row")
    return df


def fromConfig(fname, *args, **kwargs):
    saqc = SaQC(*args, **kwargs)
    config = readFile(fname)

    for _, field, expr in config.itertuples():

        regex = False
        if isQuoted(field):
            field = field[1:-1]
            regex = True

        tree = ast.parse(expr, mode="eval")
        func, kwargs = ConfigFunctionParser().parse(tree.body)

        saqc = getattr(saqc, func)(field=field, regex=regex, **kwargs)

    return saqc
