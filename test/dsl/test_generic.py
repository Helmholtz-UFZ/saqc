#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from test.common import initData

from dsl import evalExpression
from flagger import SimpleFlagger
from funcs.functions import flagGeneric, Params


def test_ismissing():

    nodata = -9999

    data = initData()
    data.iloc[:len(data)//2, 0] = np.nan
    data.iloc[(len(data)//2)+1:, 0] = nodata

    flagger = SimpleFlagger()
    flags = flagger.initFlags(data)

    var1, var2, *_ = data.columns

    idx = evalExpression("ismissing({:})".format(var1),
                         flagger,
                         data, flags,
                         var2,
                         nodata=nodata)

    fdata = data.loc[idx, var1]
    assert (pd.isnull(fdata) | (fdata == nodata)).all()


def test_isflagged():

    flagger = SimpleFlagger()
    data = initData()
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flags.iloc[::2, 0] = flagger.setFlag(flags.iloc[::2, 0])

    idx = evalExpression("isflagged({:})".format(var1),
                         flagger,
                         data, flags,
                         var2)

    flagged = flagger.isFlagged(flags[var1])
    assert (flagged == idx).all


def test_isflaggedArgument():

    flagger = SimpleFlagger()
    data = initData()
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    flags.iloc[::2, 0] = flagger.setFlag(flags.iloc[::2, 0], 1)

    idx = evalExpression("isflagged({:}, 1)".format(var1),
                         flagger,
                         data, flags,
                         var2)

    flagged = flagger.isFlagged(flags[var1], 1)
    assert (flagged == idx).all


def test_flagFailure():
    flagger = SimpleFlagger()
    data = initData()
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    # expression does not return a result of identical shape
    with pytest.raises(TypeError):
        flagGeneric(data, flags, var2, flagger,
                    **{Params.FUNC: f"sum({var1})"})

    # need a test for missing variables
    with pytest.raises(NameError):
        flagGeneric(data, flags, var2, flagger,
                    **{Params.FUNC: f"sum({var1 + 'x'})"})


if __name__ == "__main__":
    test_ismissing()
    test_isflagged()
    test_isflaggedArgument()
    test_flagFailure()
