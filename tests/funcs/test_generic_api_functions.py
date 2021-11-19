#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from dios.dios.dios import DictOfSeries

from saqc.constants import BAD, UNFLAGGED
from saqc.core.register import register
from saqc import SaQC
from saqc.lib.tools import toSequence

from tests.common import initData, flagAll


register(mask=["field"], demask=["field"], squeeze=["field"])(flagAll)


@pytest.fixture
def data():
    return initData()


def test_addFieldFlagGeneric(data):
    func = lambda x: pd.Series(False, index=x.index)
    saqc = SaQC(data=data).genericFlag(field="var1", target="tmp1", func=func, flag=BAD)
    assert (saqc.flags["tmp1"] == UNFLAGGED).all()
    assert (saqc.data["tmp1"].isna()).all()


def test_addFieldProcGeneric(data):
    saqc = SaQC(data=data)
    fields = ["var1", "var2"]
    params = [
        ("tmp", lambda x, y: x + y),
        (["tmp1", "tmp2"], lambda x, y: (x + y, y * 2)),
    ]
    for target, func in params:
        expected = DictOfSeries(
            func(*[data[f] for f in fields]), columns=toSequence(target)
        ).squeeze()
        res = saqc.genericProcess(field=fields, target=target, func=func, flag=BAD)
        assert (expected == res.data[target]).all(axis=None)
