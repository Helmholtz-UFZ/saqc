#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pytest
import numpy as np
import pandas as pd

from saqc.common import *
from saqc.flagger import initFlagsLike
from saqc.funcs import flagRange
from saqc.lib import plotting as splot
from saqc import SaQC, register

from tests.common import initData, flagAll

# no logging output needed here
# -> can this be configured on the test runner level?
logging.disable(logging.CRITICAL)


OPTIONAL = [False, True]


register(masking='field')(flagAll)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(data, optional):
    if not optional:
        return initFlagsLike(data[data.columns[::2]]).toDios()


def test_errorHandling(data):

    @register(masking='field')
    def raisingFunc(data, field, flagger, **kwargs):
        raise TypeError

    var1 = data.columns[0]

    for policy in ["ignore", "warn"]:
        # NOTE: should not fail, that's all we are testing here
        SaQC(data, error_policy=policy).raisingFunc(var1).getResult()

    with pytest.raises(TypeError):
        SaQC(data, error_policy='raise').raisingFunc(var1).getResult()


def test_duplicatedVariable():
    data = initData(1)
    var1 = data.columns[0]

    pdata, pflags = SaQC(data).flagtools.flagDummy(var1).getResult()

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

    pdata, pflagger = SaQC(data).flagAll(field=var1, target=target).getResult(raw=True)

    assert (pdata[var1] == pdata[target]).all(axis=None)
    assert all(pflagger[var1] == UNFLAGGED)
    assert all(pflagger[target] > UNFLAGGED)


@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flagger = initFlagsLike(data)
    flags = flagger.toDios()
    var1, var2 = data.columns[:2]

    pdata, pflagger = SaQC(data, flags=flags).flagAll(var1).flagAll(var2).getResult(raw=True)

    for c in pflagger.columns:
        assert pflagger[c].dtype == flagger[c].dtype


def test_plotting(data):
    """
    Test if the plotting code runs, does not show any plot.

    NOTE:
    This test is ignored if matplotlib is not available on the test-system
    """
    pytest.importorskip("matplotlib", reason="requires matplotlib")
    field, *_ = data.columns
    flagger = initFlagsLike(data)
    _, flagger_range = flagRange(data, field, flagger, min=10, max=90, flag=BAD)
    data_new, flagger_range = flagRange(data, field, flagger_range, min=40, max=60, flag=DOUBT)
    splot._interactive = False
    splot._plotSingleVariable(data, data_new, flagger, flagger_range, sources=[], targets=[data_new.columns[0]])
    splot._plotMultipleVariables(data, data_new, flagger, flagger_range, targets=data_new.columns)
    splot._interactive = True
