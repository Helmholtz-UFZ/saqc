#! /usr/bin/env python
# -*- coding: utf-8 -*-

from test.common import initData, initMeta
from core import runner
from flagger.dmpflagger import DmpFlagger, FlagFields


def test_basic():

    flagger = DmpFlagger()
    data = initData()
    var1, var2, *_ = data.columns
    var1mean = data[var1].mean()
    var2mean = data[var2].mean()

    metastring = f"""
    headerout, Flag_1, Flag_2
    {var1},"generic, {{func: this < {var1mean}, flag: DOUBTFUL}}","range, {{min: 10, max: 20, comment: saqc}}"
    {var2},"generic, {{func: this > {var2mean}, cause: error}}"
    """
    meta = initMeta(metastring, data)

    data, flags = runner(meta, flagger, data)

    col1 = data[var1]
    col2 = data[var2]

    flags11 = flags.loc[col1 < var1mean, (var1, FlagFields.FLAG)]
    flags12 = flags.loc[((col1 < 10) | (col1 > 20)), (var1, FlagFields.COMMENT)]
    # flags12 = flags.loc[(col1 >= var1mean) & ((col1 < 10) | (col1 > 20)), (var1, FlagFields.COMMENT)]

    flags21 = flags.loc[col2 > var2mean, (var2, FlagFields.CAUSE)]

    assert (flags11 > flagger.flags.min()).all()
    assert (flags12 == "saqc").all()
    assert (flags21 == "error").all()


def test_flagOrder():

    data = initData()
    var, *_ = data.columns

    flagger = DmpFlagger()
    fmin = flagger.flags.min()
    fmax = flagger.flags.max()

    metastring = f"""
    headerout,Flag
    {var},"generic, {{func: this > mean(this), flag: {fmax}}}"
    {var},"generic, {{func: this >= min(this), flag: {fmin}}}"
    """
    meta = initMeta(metastring, data)

    pdata, pflags = runner(meta, flagger, data)

    datacol = pdata[var]
    flagcol = pflags[(var, FlagFields.FLAG)]

    assert (flagcol[datacol > datacol.mean()] == fmax).all()
    assert (flagcol[datacol <= datacol.mean()] == fmin).all()


if __name__ == "__main__":

    test_basic()
    test_flagOrder()
