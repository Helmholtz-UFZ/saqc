#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np
import pandas as pd

from saqc import SaQC, register
from saqc.funcs import flagRange
from saqc.lib import plotting as splot
from test.common import initData, TESTFLAGGER, flagAll


# no logging output needed here
# -> can this be configured on the test runner level?
logging.disable(logging.CRITICAL)


OPTIONAL = [False, True]


register(masking='field')(flagAll)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(flagger, data, optional):
    if not optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_errorHandling(data, flagger):

    @register(masking='field')
    def raisingFunc(data, field, flagger, **kwargs):
        raise TypeError

    var1 = data.columns[0]

    for policy in ["ignore", "warn"]:
        # NOTE: should not fail, that's all we are testing here
        SaQC(flagger, data, error_policy=policy).raisingFunc(var1).getResult()

    with pytest.raises(TypeError):
        SaQC(flagger, data, error_policy='raise').raisingFunc(var1).getResult()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_duplicatedVariable(flagger):
    data = initData(1)
    var1 = data.columns[0]

    pdata, pflags = SaQC(flagger, data).flagDummy(var1).flagDummy(var1).getResult()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert np.all(cols == [var1])
    else:
        assert (pflags.columns == [var1]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_sourceTarget(flagger):
    """
    test implicit assignments
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    pdata, pflagger = SaQC(flagger, data).flagAll(field=var1, target=target).getResult(raw=True)
    pflags = pflagger.isFlagged()

    assert (pdata[var1] == pdata[target]).all(axis=None)
    assert (pflags[var1] == False).all(axis=None)
    assert (pflags[target] == True).all(axis=None)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flagger, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    var1, var2 = data.columns[:2]

    pdata, pflagger = SaQC(flagger, data, flags=flags).flagAll(var1).flagAll(var2).getResult(raw=True)

    pflags = pflagger.getFlags()
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_plotting(data, flagger):
    """
    Test if the plotting code runs, does not show any plot.

    NOTE:
    This test is ignored if matplotlib is not available on the test-system
    """
    pytest.importorskip("matplotlib", reason="requires matplotlib")
    field, *_ = data.columns
    flagger = flagger.initFlags(data)
    _, flagger_range = flagRange(data, field, flagger, min=10, max=90, flag=flagger.BAD)
    data_new, flagger_range = flagRange(data, field, flagger_range, min=40, max=60, flag=flagger.GOOD)
    splot._interactive = False
    splot._plotSingleVariable(data, data_new, flagger, flagger_range, sources=[], targets=[data_new.columns[0]])
    splot._plotMultipleVariables(data, data_new, flagger, flagger_range, targets=data_new.columns)
    splot._interactive = True
