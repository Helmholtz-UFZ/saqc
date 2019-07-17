#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd

from saqc.core.core import runner
from saqc.flagger.dmpflagger import DmpFlagger, FlagFields
from saqc.core.config import Fields


def test_basic():

    flagger = DmpFlagger()
    data = initData()
    var1, var2, *_ = data.columns
    var1mean = data[var1].mean()
    var2mean = data[var2].mean()

    metastring = f"""
    {Fields.VARNAME}|Flag_1                                               |Flag_2
    {var1}          |"generic,{{func: this < {var1mean}, flag: DOUBTFUL}}"|"range, {{min: 10, max: 20, comment: saqc}}"
    {var2}          |"generic,{{func: this > {var2mean}, cause: error}}"  |
    """
    metafobj, meta = initMeta(metastring, data)

    pdata, pflags = runner(metafobj, flagger, data)

    col1 = pdata[var1]
    col2 = pdata[var2]

    pflags11 = pflags.loc[col1 < var1mean, (var1, F.FLAG)]
    pflags21 = pflags.loc[col2 > var2mean, (var2, F.CAUSE)]
    pflags12 = pflags.loc[((col1 < 10) | (col1 > 20)), (var1, F.COMMENT)]
    pflags12 = pd.io.json.json_normalize(pflags12.apply(json.loads))

    assert (pflags11 > flagger.GOOD).all()
    assert set(["comment", "commit", "test"]) == set(pflags12.columns)
    assert (pflags12["comment"] == "saqc").all()
    assert (pflags21 == "error").all()


def test_flagOrder():

    data = initData()
    var, *_ = data.columns

    flagger = DmpFlagger()
    fmin = flagger.GOOD
    fmax = flagger.BAD

    metastring = f"""
    {Fields.VARNAME},Flag
    {var},"generic, {{func: this > mean(this), flag: {fmax}}}"
    {var},"generic, {{func: this >= min(this), flag: {fmin}}}"
    """
    metafobj, meta = initMeta(metastring, data)

    pdata, pflags = runner(metafobj, flagger, data)

    datacol = pdata[var]
    flagcol = pflags[(var, FlagFields.FLAG)]

    assert (flagcol[datacol > datacol.mean()] == fmax).all()
    assert (flagcol[datacol <= datacol.mean()] == fmin).all()


if __name__ == "__main__":

    test_basic()
    test_flagOrder()
