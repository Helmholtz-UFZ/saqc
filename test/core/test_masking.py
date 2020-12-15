#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytest
import pandas as pd

from saqc import SaQC, register
from test.common import initData, TESTFLAGGER


logging.disable(logging.CRITICAL)


@pytest.fixture
def data():
    return initData(3)


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

    pdata, pflagger = qc.getResult(raw=True)
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

    @register(masking='all')
    def changeData(data, field, flagger, **kwargs):
        mask = data.isna()
        data.aloc[mask] = FILLER
        return data, flagger

    @register(masking='all')
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

    @register(masking='none')
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

    data, flagger = qc.getResult(raw=True)
    assert (data[var] == FILLER).all(axis=None)
