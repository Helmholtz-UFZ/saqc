#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from ..common import initData
from saqc.flagger.simpleflagger import SimpleFlagger
from saqc.dsl.parser import evalExpression


@pytest.fixture
def data():
    return initData(3)


def test_missingIdentifier():
    data = initData()
    flagger = SimpleFlagger()
    flags = flagger.initFlags(data)
    tests = ["func(var2) < 5", "var3 != NODATA"]
    for test in tests:
        with pytest.raises(NameError):
            evalExpression(test, flagger, data, flags, data.columns[0])


def test_flagPropagation():
    data = initData()
    flagger = SimpleFlagger()
    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, 'var2', iloc=slice(None, None, 5))

    var1, var2, *_ = data.columns
    var2_flags = flagger.isFlagged(flags[var2])
    var2_data = data[var2].mask(var2_flags)

    result = evalExpression("var2 < mean(var2)",
                            flagger,
                            data, flags,
                            data.columns[0])

    expected = (var2_flags | (var2_data < var2_data.mean()))
    assert (result.filled(True) == expected).all()


def test_isflagged():
    data = initData(cols=1)
    flagger = SimpleFlagger()
    flags = flagger.initFlags(data)
    flags = flagger.setFlags(flags, 'var1', iloc=slice(None, None, 5), flag=flagger.BAD)
    flags = flagger.setFlags(flags, 'var1', iloc=slice(1, None, 5), flag=flagger.GOOD)

    tests = {
        "isflagged(this)" : flagger.isFlagged(flags, flagger.GOOD, ">"),
        f"isflagged(this, {flagger.GOOD})" : flagger.isFlagged(flags, flagger.GOOD, "=="),
        # NOTE: _ast.Str is not implemented, not sure if we should do so
        # f"isflagged(this, {flagger.GOOD}, '<')" : flagger.isFlagged(flags, flagger.GOOD, "<"),
    }
    for expr, right in tests.items():
        left = evalExpression(expr, flagger, data, flags, data.columns[0])
        assert np.all(left.to_frame() == right)
