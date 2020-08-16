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


@register(all_data=False)
def flagAll(data, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    return data, flagger.setFlags(field=field, flag=flagger.BAD)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(flagger, data, optional):
    if not optional:
        return flagger.initFlags(data[data.columns[::2]])._flags


@pytest.mark.skip(reason="does not make sense anymore")
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_errorHandling(data, flagger):

    @register(all_data=False)
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

    pdata, pflagger = SaQC(flagger, data).flagDummy(var1).flagDummy(var1).getResult()
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

    pdata, pflagger = SaQC(flagger, data).flagAll(var1).flagAll(var2).getResult()
    pflags = pflagger.getFlags()

    assert (set(pflags.columns) == {var1, var2})
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

    pdata, pflagger = SaQC(flagger, data, flags=flags).flagAll(var1).flagAll(var2).getResult()

    pflags = pflagger.getFlags()
    assert dict(flags.dtypes) == dict(pflags.dtypes)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_masking(data, flagger):
    """
    test if flagged values are exluded during the preceding tests
    """
    flagger = flagger.initFlags(data)
    var1 = 'var1'
    mn = min(data[var1])
    mx = max(data[var1]) / 2

    qc = SaQC(flagger, data)
    qc = qc.flagRange(var1, mn, mx)
    # min is not considered because its the smalles possible value.
    # if masking works, `data > max` will be masked,
    # so the following will deliver True for in range (data < max),
    # otherwise False, like an inverse range-test
    qc = qc.procGeneric("dummy", func=lambda var1: var1 >= mn)

    pdata, pflagger = qc.getResult()
    out_of_range = pflagger.isFlagged(var1)
    in_range = ~out_of_range

    assert (pdata.loc[out_of_range, "dummy"] == False).all()
    assert (pdata.loc[in_range, "dummy"] == True).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_masking_UnmaskingOnDataChange(data, flagger):
    """ test if (un)masking works as expected on data-change.

    If the data change in the func, unmasking should respect this changes and
    should not reapply original data, instead take the new data (and flags) as is.
    Also if flags change, the data should be taken as is.
    """
    FILLER = -9999

    @register
    def changeData(data, field, flagger, **kwargs):
        mask = data.isna()
        data.aloc[mask] = FILLER
        return data, flagger

    @register
    def changeFlags(data, field, flagger, **kwargs):
        mask = data.isna()
        flagger = flagger.setFlags(field, loc=mask[field], flag=flagger.UNFLAGGED, force=True)
        return data, flagger

    var = data.columns[0]
    var_data = data[var]
    mn, mx = var_data.max() * .25, var_data.max() * .75
    range_mask = (var_data < mn) | (var_data > mx)

    qc = SaQC(flagger, data)
    qc = qc.flagRange(var, mn, mx)
    qcD = qc.changeData(var)
    qcF = qc.changeFlags(var)

    data, flagger = qcD.getResult()
    assert (data[var][range_mask] == FILLER).all(axis=None)
    # only flags change so the data should be still NaN, because
    # the unmasking was disabled, but the masking indeed was happening
    data, flagger = qcF.getResult()
    assert data[var][range_mask].isna().all(axis=None)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_shapeDiffUnmasking(data, flagger):
    """ test if (un)masking works as expected on index-change.

    If the index of data (and flags) change in the func, the unmasking,
    should not reapply original data, instead take the new data (and flags) as is.
    """

    FILLER = -1111

    @register
    def pseudoHarmo(data, field, flagger, **kwargs):
        index = data[field].index.to_series()
        index.iloc[-len(data[field])//2:] += pd.Timedelta("7.5Min")

        data[field] = pd.Series(data=FILLER, index=index)

        flags = flagger.getFlags()
        flags[field] = pd.Series(data=flags[field].values, index=index)

        flagger = flagger.initFlags(flags=flags)
        return data, flagger

    var = data.columns[0]
    var_data = data[var]
    mn, mx = var_data.max() * .25, var_data.max() * .75

    qc = SaQC(flagger, data)
    qc = qc.flagRange(var, mn, mx)
    qc = qc.pseudoHarmo(var)

    data, flagger = qc.getResult()
    assert (data[var] == FILLER).all(axis=None)


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
