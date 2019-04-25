#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from test.common import initData
from core import runner, prepareMeta
from flagger.dmpflagger import DmpFlagger, FlagFields


def test_DmpFlagger():

    data = initData()
    var1, var2, *_ = data.columns
    var1mean = data[var1].mean()
    var2mean = data[var2].mean()

    meta = [
        [var1, f"generic, {{func: this < {var1mean}}}", "range, {min: 10, max: 20, comment: saqc}"],
        [var2, f"generic, {{func: this > {var2mean}, cause: error}}"],
    ]
    meta = prepareMeta(
        pd.DataFrame(meta, columns=["headerout", "Flag_1", "Flag_2"]),
        data)

    flagger = DmpFlagger()
    data, flags = runner(meta, flagger, data)

    col1 = data[var1]
    col2 = data[var2]

    flags11 = flags.loc[col1 < var1mean, (var1, FlagFields.FLAG)]
    flags12 = flags.loc[((col1 < 10) | (col1 > 20)), (var1, FlagFields.COMMENT)]

    flags21 = flags.loc[col2 > var2mean, (var2, FlagFields.CAUSE)]

    assert (flags11 >= flagger.flags.min()).all()
    assert (flags12 == "saqc").all()
    assert (flags21 == "error").all()


def test_flagOrder():

    data = initData()
    var, *_ = data.columns

    flagger = DmpFlagger()
    fmin = flagger.flags.min()
    fmax = flagger.flags.max()

    meta = [
        [var, f"generic, {{func: this > mean(this), flag: {fmax}}}"],
        [var, f"generic, {{func: this >= min(this), flag: {fmin}}}"],
    ]

    meta = prepareMeta(
        pd.DataFrame(meta, columns=["headerout", "Flag_1"]),
        data)

    pdata, pflags = runner(meta, flagger, data)

    datacol = pdata[var]
    flagcol = pflags[(var, FlagFields.FLAG)]

    assert (flagcol[datacol > datacol.mean()] == fmax).all()
    assert (flagcol[datacol <= datacol.mean()] == fmin).all()


if __name__ == "__main__":

    test_DmpFlagger()
    test_flagOrder()
