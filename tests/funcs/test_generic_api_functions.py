#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.constants import *
from saqc.core.register import register
from saqc.funcs.tools import maskTime
from saqc import SaQC

from tests.common import initData, flagAll


register(datamask="field")(flagAll)


@pytest.fixture
def data():
    return initData()


def test_addFieldFlagGeneric(data):
    saqc = SaQC(data=data)

    func = lambda var1: pd.Series(False, index=data[var1.name].index)
    result = saqc.genericFlag("tmp1", func, flag=BAD).result
    assert "tmp1" in result.flags.columns and "tmp1" not in result.data


def test_addFieldProcGeneric(data):
    saqc = SaQC(data=data)

    func = lambda: pd.Series([], dtype=float)
    data = saqc.genericProcess("tmp1", func).result.dataRaw
    assert "tmp1" in data.columns and data["tmp1"].empty

    func = lambda var1, var2: var1 + var2
    data = saqc.genericProcess("tmp2", func).result.dataRaw
    assert "tmp2" in data.columns and (data["tmp2"] == data["var1"] + data["var2"]).all(
        axis=None
    )
