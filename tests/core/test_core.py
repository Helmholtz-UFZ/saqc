#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

import saqc
from saqc.constants import *
from saqc.core import initFlagsLike
from saqc import SaQC, register

from tests.common import initData, flagAll

OPTIONAL = [False, True]


register(datamask="field")(flagAll)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(data, optional):
    if not optional:
        return initFlagsLike(data[data.columns[::2]]).toDios()


def test_errorHandling(data):
    @register(datamask="field")
    def raisingFunc(data, field, flags, **kwargs):
        raise TypeError

    var1 = data.columns[0]

    with pytest.raises(TypeError):
        SaQC(data).raisingFunc(var1).getResult()


def test_duplicatedVariable():
    data = initData(1)
    var1 = data.columns[0]

    _, pflags = SaQC(data).flagDummy(var1).getResult()

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = pflags.columns.get_level_values(0).drop_duplicates()
        assert np.all(cols == [var1])
    else:
        assert (pflags.columns == [var1]).all()


@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flags = initFlagsLike(data)
    flags_raw = flags.toDios()
    var1, var2 = data.columns[:2]

    _, pflags = (
        SaQC(data, flags=flags_raw).flagAll(var1).flagAll(var2).getResult(raw=True)
    )

    for c in pflags.columns:
        assert pflags[c].dtype == flags[c].dtype


def test_copy(data):
    qc = saqc.SaQC(data)

    qc = qc.flagRange("var1").flagRange("var1", min=0, max=0)

    deep = qc.copy(deep=True)
    shallow = qc.copy(deep=False)

    for copy in [deep, shallow]:
        assert copy is not qc
        assert copy.called is not qc.called
        assert copy._translator is not qc._translator
        assert copy._attrs is not qc._attrs

        assert copy._data is not qc._data
        assert copy._flags is not qc._flags

        assert copy._data._data is not qc._data._data
        assert copy._flags._data is not qc._flags._data

    # underling data copied
    assert deep._data._data.iloc[0] is not qc._data._data.iloc[0]
    assert (
        deep._flags._data["var1"].hist.index is not qc._flags._data["var1"].hist.index
    )

    # underling data NOT copied
    assert shallow._data._data.iloc[0] is qc._data._data.iloc[0]
    assert shallow._flags._data["var1"].hist.index is qc._flags._data["var1"].hist.index
