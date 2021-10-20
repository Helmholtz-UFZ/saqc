#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd


from saqc.constants import *
from saqc.core import initFlagsLike
from saqc import SaQC, flagging

from tests.common import initData, flagAll


OPTIONAL = [False, True]


flagging(masking="field")(flagAll)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(data, optional):
    if not optional:
        return initFlagsLike(data[data.columns[::2]]).toDios()


def test_errorHandling(data):
    @flagging(masking="field")
    def raisingFunc(data, field, flags, **kwargs):
        raise TypeError

    var1 = data.columns[0]

    with pytest.raises(TypeError):
        SaQC(data).raisingFunc(var1).getResult()


def test_duplicatedVariable():
    data = initData(1)
    var1 = data.columns[0]

    pdata, pflags = SaQC(data).flagDummy(var1).getResult()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert np.all(cols == [var1])
    else:
        assert (pflags.columns == [var1]).all()


def test_sourceTarget():
    """
    test implicit assignments
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    pdata, pflags = SaQC(data).flagAll(field=var1, target=target).getResult(raw=True)

    assert (pdata[var1] == pdata[target]).all(axis=None)
    assert all(pflags[var1] == UNFLAGGED)
    assert all(pflags[target] > UNFLAGGED)


@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flags = initFlagsLike(data)
    flags_raw = flags.toDios()
    var1, var2 = data.columns[:2]

    pdata, pflags = (
        SaQC(data, flags=flags_raw).flagAll(var1).flagAll(var2).getResult(raw=True)
    )

    for c in pflags.columns:
        assert pflags[c].dtype == flags[c].dtype
