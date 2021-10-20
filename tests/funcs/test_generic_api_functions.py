#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.constants import *
from saqc.core.register import flagging
from saqc.funcs.tools import maskTime
from saqc import SaQC

from tests.common import initData, flagAll


flagging(masking="field")(flagAll)


@pytest.fixture
def data():
    return initData()


def test_addFieldFlagGeneric(data):
    saqc = SaQC(data=data)

    func = lambda var1: pd.Series(False, index=data[var1.name].index)
    data, flags = saqc.genericFlag("tmp1", func, flag=BAD).getResult()
    assert "tmp1" in flags.columns and "tmp1" not in data


def test_addFieldProcGeneric(data):
    saqc = SaQC(data=data)

    func = lambda: pd.Series([])
    data, flags = saqc.genericProcess("tmp1", func, flag=BAD).getResult(raw=True)
    assert "tmp1" in data.columns and data["tmp1"].empty

    func = lambda var1, var2: var1 + var2
    data, flags = saqc.genericProcess("tmp2", func, flag=BAD).getResult()
    assert "tmp2" in data.columns and (data["tmp2"] == data["var1"] + data["var2"]).all(
        axis=None
    )
