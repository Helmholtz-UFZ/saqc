#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd

from saqc.constants import *
from saqc.core.register import flagging
from saqc.funcs.tools import mask
from saqc import SaQC

from tests.common import initData, flagAll


flagging(masking="field")(flagAll)


@pytest.fixture
def data():
    return initData()


def test_addFieldFlagGeneric(data):
    saqc = SaQC(data=data)

    func = lambda var1: pd.Series(False, index=data[var1.name].index)
    data, flags = saqc.generic.flag("tmp1", func, flag=BAD).getResult()
    assert "tmp1" in flags.columns and "tmp1" not in data


def test_addFieldProcGeneric(data):
    saqc = SaQC(data=data)

    func = lambda: pd.Series([])
    data, flags = saqc.generic.process("tmp1", func, flag=BAD).getResult(raw=True)
    assert "tmp1" in data.columns and data["tmp1"].empty

    func = lambda var1, var2: var1 + var2
    data, flags = saqc.generic.process("tmp2", func, flag=BAD).getResult()
    assert "tmp2" in data.columns and (data["tmp2"] == data["var1"] + data["var2"]).all(
        axis=None
    )


def test_mask(data):
    saqc = SaQC(data=data)
    data_org = data.copy(deep=True)
    mean = data["var1"] / 2

    data, _ = saqc.generic.process(
        "var1", lambda var1: mask(var1 < mean), flag=BAD
    ).getResult()
    assert (
        (data["var1"].isna()) == (data_org["var1"] < 10) & data_org["var1"].isna()
    ).all(axis=None)

    data, flags = saqc.generic.process(
        "tmp", lambda var1: mask(var1 < mean), flag=BAD
    ).getResult()
    assert ("tmp" in data.columns) and ("tmp" in flags.columns)
    assert (
        (data["tmp"].isna()) == (data_org["var1"] < 10) & data_org["var1"].isna()
    ).all(axis=None)
