#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from test.common import initData
from core import runner, prepareMeta
from flagger.dmpflagger import DmpFlagger, FlagFields, Flags


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

    data, flags = runner(meta, DmpFlagger(), data)

    col1 = data[var1]
    col2 = data[var2]

    flags11 = flags.loc[col1 < var1mean, (var1, FlagFields.FLAG)]
    flags12 = flags.loc[((col1 < 10) | (col1 > 20)), (var1, FlagFields.COMMENT)]

    flags21 = flags.loc[col2 > var2mean, (var2, FlagFields.CAUSE)]

    assert (flags11 == Flags.BAD).all()
    assert (flags12 == "saqc").all()
    assert (flags21 == "error").all()





if __name__ == "__main__":

    test_DmpFlagger()
