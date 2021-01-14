#! /usr/bin/env python
# -*- coding: utf-8 -*-
import ast

import pytest
import numpy as np
import pandas as pd

from dios import DictOfSeries

from test.common import TESTFLAGGER, TESTNODATA, initData, writeIO, flagAll
from saqc.core.visitor import ConfigFunctionParser
from saqc.core.config import Fields as F
from saqc.core.register import register
from saqc import SaQC, SimpleFlagger
from saqc.funcs.functions import _execGeneric


register(masking='field')(flagAll)


@pytest.fixture
def data():
    return initData()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_addFieldFlagGeneric(data, flagger):
    saqc = SaQC(data=data, flagger=flagger)

    data, flags = saqc.flagGeneric(
        "tmp1",
        func=lambda var1: pd.Series(False, index=data[var1.name].index)
    ).getResult()
    assert "tmp1" in flags.columns and "tmp1" not in data


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_addFieldProcGeneric(data, flagger):
    saqc = SaQC(data=data, flagger=flagger)

    data, flagger = saqc.procGeneric("tmp1", func=lambda: pd.Series([])).getResult(raw=True)
    assert "tmp1" in data.columns and data["tmp1"].empty

    data, flagger = saqc.procGeneric("tmp2", func=lambda var1, var2: var1 + var2).getResult()
    assert "tmp2" in data.columns and (data["tmp2"] == data["var1"] + data["var2"]).all(axis=None)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mask(data, flagger):

    saqc = SaQC(data=data, flagger=flagger)
    data_org = data.copy(deep=True)
    mean = data["var1"] / 2

    data, _ = saqc.procGeneric("var1", lambda var1: mask(var1 < mean)).getResult()
    assert ((data["var1"].isna()) == (data_org["var1"] < 10) & data_org["var1"].isna()).all(axis=None)

    data, flags = saqc.procGeneric("tmp", lambda var1: mask(var1 < mean)).getResult()
    assert ("tmp" in data.columns) and ("tmp" in flags.columns)
    assert ((data["tmp"].isna()) == (data_org["var1"] < 10) & data_org["var1"].isna()).all(axis=None)
