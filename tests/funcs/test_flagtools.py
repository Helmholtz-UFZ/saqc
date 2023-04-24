#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import operator

import numpy as np
import pandas as pd
import pytest

from saqc import BAD as B
from saqc import UNFLAGGED as U
from saqc import SaQC
from saqc.funcs.flagtools import _groupOperation

N = np.nan


@pytest.mark.parametrize(
    "got, expected, kwargs",
    [
        ([N, N, B, N, N], [N, N, N, B, N], {"window": 1, "method": "ffill"}),
        ([N, N, B, N, N], [N, B, N, N, N], {"window": 1, "method": "bfill"}),
        ([B, N, N, N, B], [N, B, N, N, N], {"window": 1, "method": "ffill"}),
        ([B, N, N, N, B], [N, N, N, B, N], {"window": 1, "method": "bfill"}),
        ([N, N, B, N, N], [N, N, N, B, N], {"window": "1D", "method": "ffill"}),
        ([N, N, B, N, N], [N, B, N, N, N], {"window": "1D", "method": "bfill"}),
        ([B, N, N, N, B], [N, B, N, N, N], {"window": "1D", "method": "ffill"}),
        ([B, N, N, N, B], [N, N, N, B, N], {"window": "1D", "method": "bfill"}),
        ([N, N, B, N, N], [N, N, N, B, B], {"window": 2, "method": "ffill"}),
        ([N, N, B, N, N], [B, B, N, N, N], {"window": 2, "method": "bfill"}),
        ([B, N, N, N, B], [N, B, B, N, N], {"window": 2, "method": "ffill"}),
        ([B, N, N, N, B], [N, N, B, B, N], {"window": 2, "method": "bfill"}),
        ([N, N, B, N, N], [N, N, N, B, B], {"window": "2D", "method": "ffill"}),
        ([N, N, B, N, N], [B, B, N, N, N], {"window": "2D", "method": "bfill"}),
        ([B, N, N, N, B], [N, B, B, N, N], {"window": "2D", "method": "ffill"}),
        ([B, N, N, N, B], [N, N, B, B, N], {"window": "2D", "method": "bfill"}),
        # window larger then data
        ([U, U, B, U, U], [N, N, N, B, B], {"window": 10, "method": "ffill"}),
        ([U, U, B, U, U], [B, B, N, N, N], {"window": 10, "method": "bfill"}),
        ([B, U, U, U, U], [N, B, B, B, B], {"window": "10D", "method": "ffill"}),
        ([B, U, U, U, U], [N, N, N, N, N], {"window": "10D", "method": "bfill"}),
        # playing with dfilter
        (
            [1, 1, B, 1, 1],
            [N, B, B, B, B],
            {"window": 2, "method": "ffill", "dfilter": 0},
        ),
        (
            [1, 1, B, 1, 1],
            [B, B, B, B, N],
            {"window": 2, "method": "bfill", "dfilter": 0},
        ),
        (
            [B, 1, 1, 1, 1],
            [N, B, B, B, B],
            {"window": "2D", "method": "ffill", "dfilter": 0},
        ),
        (
            [B, 1, 1, 1, 1],
            [B, B, B, B, N],
            {"window": "2D", "method": "bfill", "dfilter": 0},
        ),
    ],
)
def test_propagateFlagsRegularIndex(got, expected, kwargs):
    index = pd.date_range("2000-01-01", periods=len(got))
    flags = pd.DataFrame({"x": got}, index=index)
    expected = pd.Series(expected, index=index)
    data = pd.DataFrame({"x": np.nan}, index=index)
    saqc = SaQC(data=data, flags=flags).propagateFlags(field="x", **kwargs)
    result = saqc._flags.history["x"].hist[1].astype(float)
    assert result.equals(expected)


@pytest.mark.parametrize(
    "got, expected, kwargs",
    [
        ([N, N, B, N, N], [N, N, N, N, N], {"window": "1D", "method": "ffill"}),
        ([N, N, B, N, N], [N, N, N, N, N], {"window": "1D", "method": "bfill"}),
        ([B, N, N, N, B], [N, B, N, N, N], {"window": "1D", "method": "ffill"}),
        ([B, N, N, N, B], [N, N, N, N, N], {"window": "1D", "method": "bfill"}),
        ([N, N, B, N, N], [N, N, N, B, N], {"window": "3D", "method": "ffill"}),
        ([N, N, B, N, N], [B, B, N, N, N], {"window": "3D", "method": "bfill"}),
        ([B, N, N, N, B], [N, B, N, N, N], {"window": "2D", "method": "ffill"}),
        ([B, N, N, N, B], [N, N, N, N, N], {"window": "2D", "method": "bfill"}),
        ([B, U, U, U, U], [N, B, B, B, N], {"window": "10D", "method": "ffill"}),
    ],
)
def test_propagateFlagsIrregularIndex(got, expected, kwargs):
    index = pd.to_datetime(
        ["2000-01-01", "2000-01-02", "2000-01-04", "2000-01-07", "2000-01-18"]
    )
    flags = pd.DataFrame({"x": got}, index=index)
    expected = pd.Series(expected, index=index)
    data = pd.DataFrame({"x": np.nan}, index=index)
    saqc = SaQC(data=data, flags=flags).propagateFlags(field="x", **kwargs)
    result = saqc._flags.history["x"].hist[1].astype(float)
    assert result.equals(expected)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([B, U, U, B], [B, B, U, U], [B, U, U, U]),
        ([B, B, B, B], [B, B, B, B], [B, B, B, B]),
        ([U, U, U, U], [U, U, U, U], [U, U, U, U]),
    ],
)
def test_andGroup(left, right, expected):
    data = pd.DataFrame({"data": [1, 2, 3, 4]})

    base = SaQC(data=data)
    this = SaQC(data=data, flags=pd.DataFrame({"data": pd.Series(left)}))
    that = SaQC(data=data, flags=pd.DataFrame({"data": pd.Series(right)}))
    result = base.andGroup(field="data", group=[this, that])

    assert pd.Series(expected).equals(result.flags["data"])


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([B, U, U, B], [B, B, U, U], [B, B, U, B]),
        ([B, B, B, B], [B, B, B, B], [B, B, B, B]),
        ([U, U, U, U], [U, U, U, U], [U, U, U, U]),
    ],
)
def test_orGroup(left, right, expected):
    data = pd.DataFrame({"data": [1, 2, 3, 4]})

    base = SaQC(data=data)
    this = SaQC(data=data, flags=pd.DataFrame({"data": pd.Series(left)}))
    that = SaQC(data=data, flags=pd.DataFrame({"data": pd.Series(right)}))
    result = base.orGroup(field="data", group=[this, that])

    assert pd.Series(expected).equals(result.flags["data"])


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([B, U, U, B], [B, B, U, U], [B, B, U, B]),
        ([B, B, B, B], [B, B, B, B], [B, B, B, B]),
        ([U, U, U, U], [U, U, U, U], [U, U, U, U]),
    ],
)
def test__groupOperationUnivariate(left, right, expected):
    data = pd.DataFrame(
        {"x": [0, 1, 2, 3], "y": [0, 11, 22, 33], "z": [0, 111, 222, 333]}
    )
    base = SaQC(data=data)
    this = SaQC(
        data=data, flags=pd.DataFrame({k: pd.Series(left) for k in data.columns})
    )
    that = SaQC(
        data=data, flags=pd.DataFrame({k: pd.Series(right) for k in data.columns})
    )
    result = _groupOperation(
        base=base, field="x", func=operator.or_, group={this: "y", that: ["y", "z"]}
    )

    assert pd.Series(expected).equals(result.flags["x"])


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (pd.Series([B, U, U, B]), pd.Series([B, B, U, U]), pd.Series([B, B, U, B])),
        (pd.Series([B, B, B, B]), pd.Series([B, B, B, B]), pd.Series([B, B, B, B])),
        (pd.Series([U, U, U, U]), pd.Series([U, U, U, U]), pd.Series([U, U, U, U])),
    ],
)
def test__groupOperationMultivariate(left, right, expected):
    data = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 11, 22, 33]})
    flags = pd.DataFrame({"x": pd.Series(left), "y": pd.Series(right)})

    qc = SaQC(data=data, flags=flags)

    # multi fields, no target
    result = _groupOperation(base=qc.copy(), field=["x", "y"], func=operator.or_)
    for v in ["x", "y"]:
        assert expected.equals(result.flags[v])

    # multi fields, multi target
    result = _groupOperation(
        base=qc.copy(), target=["a", "b"], field=["x", "y"], func=operator.or_
    )
    for v in ["a", "b"]:
        assert expected.equals(result.flags[v])
    for v, e in zip(["x", "y"], [left, right]):
        assert e.equals(result.flags[v])

    # multi fields, single target
    result = _groupOperation(
        base=qc.copy(), target="a", field=["x", "y"], func=operator.or_
    )
    assert expected.equals(result.flags["a"])
    assert result.data["a"].isna().all()
    for v, e in zip(["x", "y"], [left, right]):
        assert e.equals(result.flags[v])
