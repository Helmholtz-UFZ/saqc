#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .testfuncs import initData
from flagger import SimpleFlagger
from dsl.evaluator import evalCondition


def testConditions():
    data = initData()
    flags = SimpleFlagger().emptyFlags(data)

    tests = [
        ("this > 100",
         data["var1"] > 100),
        ("var2 < 100",
         data["var2"] < 100),
        ("abs(var2 - var1)/2 > 100",
         abs(data["var2"] - data["var1"])/2 > 100),
        ("mean(var2) > max(var1)",
         np.mean(data["var2"]) > np.max(data["var1"])),
        ("sum(var2)/len(var2) > max(var1)",
         np.mean(data["var2"]) > np.max(data["var1"]))]

    for test, expected in tests:
        idx = evalCondition(test, data, flags, data.columns[0])
        assert (idx == expected).all()


def testMissingIdentifier():
    data = initData()
    flags = SimpleFlagger().emptyFlags(data)
    tests = ["func(var2) < 5", "var3 != NODATA"]
    for test in tests:
        try:
            evalCondition(test, data, flags, data.columns[0])
        except NameError:
            continue
        else:
            raise AssertionError


if __name__ == "__main__":
    testConditions()
    testMissingIdentifier()
