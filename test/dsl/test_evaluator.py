#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from ..common import initData
from saqc.flagger.simpleflagger import SimpleFlagger
from saqc.dsl.evaluator import evalExpression


def test_evaluationBool():
    data = initData()
    flagger = SimpleFlagger()
    flags = flagger.initFlags(data)
    var1, var2, *_ = data.columns

    tests = [
        ("this > 100",
         data[var1] > 100),
        ("var2 < 100",
         data[var2] < 100),
        (f"abs({var2} - {var1})/2 > 100",
         abs(data[var2] - data[var1])/2 > 100),
        (f"mean({var2}) > max({var1})",
         np.mean(data[var2]) > np.max(data[var1])),
        (f"sum({var2})/len({var2}) > max({var1})",
         np.mean(data[var2]) > np.max(data[var1]))]

    for test, expected in tests:
        result = evalExpression(test, flagger, data, flags, data.columns[0])
        if isinstance(result, np.ma.MaskedArray):
            result = result.filled(True)
        assert (result == expected).all()


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
    flags.iloc[::5] = flagger.setFlag(flags.iloc[::5])

    var1, var2, *_ = data.columns
    var2_flags = flagger.isFlagged(flags[var2])
    var2_data = data[var2].mask(var2_flags)

    result = evalExpression("var2 < mean(var2)",
                            flagger,
                            data, flags,
                            data.columns[0])

    expected = (var2_flags | (var2_data < var2_data.mean()))
    assert (result.filled(True) == expected).all()


if __name__ == "__main__":
    test_evaluationBool()
    test_missingIdentifier()
    test_flagPropagation()
