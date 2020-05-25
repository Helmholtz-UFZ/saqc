#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np
import pandas as pd

from saqc import SaQC, register
from saqc.funcs import flagRange
from saqc.lib import plotting as splot
from test.common import initData, TESTFLAGGER


# no logging output needed here
# -> can this be configured on the test runner level?
logging.disable(logging.CRITICAL)


OPTIONAL = [False, True]


@register
def flagAll(data, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    return data, flagger.setFlags(field=field, flag=flagger.BAD)


@pytest.fixture
def data():
    return initData(3)


def _initFlags(flagger, data, optional):
    return None
    if optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


@pytest.fixture
def flags(flagger, data, optional):
    if not optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


@pytest.mark.skip(reason="does not make sense anymore")
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_errorHandling(data, flagger):

    @register
    def raisingFunc(data, field, flagger, **kwargs):
        raise TypeError

    var1 = data.columns[0]

    for policy in ["ignore", "warn"]:
        # NOTE: should not fail, that's all we are testing here
        SaQC(flagger, data, error_policy=policy).raisingFunc(var1).getResult()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_duplicatedVariable(flagger):
    data = initData(1)
    var1 = data.columns[0]

    pdata, pflagger = (SaQC(flagger, data)
                       .flagDummy(var1)
                       .flagDummy(var1)
                       .getResult())
    pflags = pflagger.getFlags()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert np.all(cols == [var1])
    else:
        assert (pflags.columns == [var1]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_assignVariable(flagger):
    """
    test implicit assignments
    """
    data = initData(1)
    var1 = data.columns[0]
    var2 = "empty"

    pdata, pflagger = (SaQC(flagger, data)
                       .flagAll(var1)
                       .flagAll(var2)
                       .getResult())
    pflags = pflagger.getFlags()

    assert (pflags.columns == [var1, var2]).all()
    assert pflagger.isFlagged(var2).empty


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flagger, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    var1, var2 = data.columns[:2]

    pdata, pflagger = (SaQC(flagger, data, flags=flags)
                       .flagAll(var1)
                       .flagAll(var2)
                       .getResult())

    pflags = pflagger.getFlags()
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_nanInjections(data, flagger):
    """
    test if flagged values are exluded during the preceding tests
    """
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    var = data.columns[0]
    mn = min(data[var])
    mx = max(data[var])/2

    pdata, pflagger = (SaQC(flagger, data, flags=flags)
                       .flagRange(var, mn, mx)
                       .procGeneric("dummy", func=lambda var1: var1 >= mn)
                       .getResult())
    assert not pdata.loc[pflagger.isFlagged(var), "dummy"].any()
    assert pdata.loc[~pflagger.isFlagged(var), "dummy"].all()


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
