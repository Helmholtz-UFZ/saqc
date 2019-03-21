#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .testfuncs import initData

from dsl import evalCondition
from flagger import SimpleFlagger


def test_ismissing():

    nodata = -9999

    data = initData()
    data.iloc[:len(data)//2, 0] = np.nan
    data.iloc[(len(data)//2)+1:, 0] = nodata

    flagger = SimpleFlagger()
    flags = flagger.emptyFlags(data)

    var1, var2, *_ = data.columns

    idx = evalCondition(
        "ismissing({:})".format(var1),
        flagger,
        data, flags,
        var2,
        nodata=nodata)

    fdata = data.loc[idx, var1]
    assert (pd.isnull(fdata) | (fdata == nodata)).all()


def test_isflagged():

    flagger = SimpleFlagger()
    data = initData()
    flags = flagger.emptyFlags(data, 0)
    var1, var2, *_ = data.columns

    flags.iloc[::2, 0] = flagger.setFlag(flags.iloc[::2, 0])

    idx = evalCondition(
        "isflagged({:})".format(var1),
        flagger,
        data, flags,
        var2)

    flagged = flagger.isFlagged(flags[var1])
    assert (flagged == idx).all


def test_isflagged_nonstandard():

    flagger = SimpleFlagger()
    data = initData()
    flags = flagger.emptyFlags(data, 0)
    var1, var2, *_ = data.columns

    flags.iloc[::2, 0] = flagger.setFlag(flags.iloc[::2, 0], -9)

    idx = evalCondition(
        "isflagged({:}, -9)".format(var1),
        flagger,
        data, flags,
        var2)

    flagged = flagger.isFlagged(flags[var1], -9)
    assert (flagged == idx).all


if __name__ == "__main__":
    test_ismissing()
    test_isflagged()
    test_isflagged_nonstandard()
