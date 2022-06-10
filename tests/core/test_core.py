#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import copy

import numpy as np
import pandas as pd
import pytest

import saqc
from saqc.core import SaQC, initFlagsLike, register
from saqc.core.flags import Flags
from saqc.core.register import flagging, processing
from tests.common import flagAll, initData

OPTIONAL = [False, True]

register(mask=["field"], demask=["field"], squeeze=["field"])(flagAll)


@pytest.fixture
def data():
    return initData(3)


@pytest.fixture
def flags(data, optional):
    if not optional:
        return initFlagsLike(data[data.columns[::2]]).toDios()


def test_errorHandling(data):
    @processing()
    def raisingFunc(data, field, flags, **kwargs):
        raise TypeError

    var1 = data.columns[0]
    qc = SaQC(data)

    with pytest.raises(TypeError):
        qc.raisingFunc(var1)


@pytest.mark.parametrize("optional", OPTIONAL)
def test_dtypes(data, flags):
    """
    Test if the categorical dtype is preserved through the core functionality
    """
    flags = initFlagsLike(data)
    flags_raw = flags.toDios()
    var1, var2 = data.columns[:2]

    pflags = SaQC(data, flags=flags_raw).flagAll(var1).flagAll(var2).flags

    for c in pflags.columns:
        assert pflags[c].dtype == flags[c].dtype


def test_copy(data):
    qc = saqc.SaQC(data)

    qc = qc.flagRange("var1").flagRange("var1", min=0, max=0)

    deep = qc.copy(deep=True)
    shallow = qc.copy(deep=False)

    for copy in [deep, shallow]:
        assert copy is not qc
        assert copy._scheme is not qc._scheme
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


def test_sourceTargetCopy():
    """
    test implicit copies
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    @register(mask=["field"], demask=["field"], squeeze=["field"], handles_target=False)
    def flagTarget(data, field, flags, **kwargs):
        assert "target" not in kwargs
        return data, flags

    qc = SaQC(data, flags=Flags({var1: pd.Series(127.0, index=data[var1].index)}))
    qc = qc.flagTarget(field=var1, target=target)

    assert (qc.data[var1] == qc.data[target]).all(axis=None)
    assert all(qc.flags[var1] == qc.flags[target])


def test_sourceTargetNoCopy():
    """
    test bypassing of the imlpicit copy machiners
    """
    data = initData(1)
    var1 = data.columns[0]
    target = "new"

    @register(mask=["field"], demask=["field"], squeeze=["field"], handles_target=True)
    def flagField(data, field, flags, **kwargs):
        assert "target" in kwargs
        assert "target" not in data
        assert "target" not in flags
        return data, flags

    SaQC(data).flagField(field=var1, target=target)


def test_sourceTargetMultivariate():
    """
    test bypassing of the imlpicit copy machiners
    """
    data = initData(3)

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        handles_target=True,
        multivariate=True,
    )
    def flagMulti(data, field, flags, **kwargs):
        assert "target" in kwargs
        assert "target" not in data
        assert "target" not in flags
        assert field == kwargs["target"]
        return data, flags

    SaQC(data).flagMulti(field=data.columns, target=data.columns)


def test_sourceTargetMulti():
    data = initData(3)
    flags = initFlagsLike(data)
    fields = data.columns
    targets = [f"target{i + 1}" for i in range(len(fields))]

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        handles_target=False,
        multivariate=True,
    )
    def flagMulti(data, field, flags, target, **kwargs):
        assert len(field) == len(target)
        for src, trg in zip(field, target):
            assert src in data
            assert trg in data
            assert src in flags
            assert trg in flags

            assert (data[src] == data[trg]).all(axis=None)
            assert (flags[src] == flags[trg]).all(axis=None)
        return data, flags

    SaQC(data, flags).flagMulti(field=fields, target=targets)


def test_unknown_attribute():
    qc = SaQC()
    with pytest.raises(AttributeError):
        qc._construct(_spam="eggs")


def test_validation(data):
    """Test if validation detects different columns in data and flags."""
    df = pd.DataFrame(
        data=np.arange(8).reshape(4, 2),
        index=pd.date_range("2020", None, 4, "1d"),
        columns=list("ab"),
    )
    qc = SaQC(df)

    @flagging()
    def flagFoo(data, field, flags, **kwargs):
        data["spam"] = data[field]
        return data, flags

    with pytest.raises(RuntimeError):
        qc.flagFoo("a")


@pytest.mark.skip(reason="bug in register, see #GL 342")
def test_validation_flags(data):
    """Test if validation detects different columns in data and flags."""
    df = pd.DataFrame(
        data=np.arange(8).reshape(4, 2),
        index=pd.date_range("2020", None, 4, "1d"),
        columns=list("ab"),
    )
    qc = SaQC(df)

    @flagging()
    def flagFoo(data, field, flags, **kwargs):
        flags["spam"] = flags[field]
        return data, flags

    with pytest.raises(RuntimeError):
        qc.flagFoo("a")


def test__copy__():
    orig = SaQC()
    orig.attrs["spam"] = []  # a higher object
    shallow = copy.copy(orig)
    assert shallow is not orig
    assert shallow.attrs["spam"] is orig.attrs["spam"]


def test__deepcopy__():
    orig = SaQC()
    orig.attrs["spam"] = []  # a higher object
    shallow = copy.deepcopy(orig)
    assert shallow is not orig
    assert shallow.attrs["spam"] is not orig.attrs["spam"]


def test_immutability(data):
    field = data.columns[0]
    saqc_before = SaQC(data)
    saqc_after = saqc_before.flagDummy(field)
    for name in SaQC._attributes:
        assert getattr(saqc_before, name) is not getattr(saqc_after, name)
