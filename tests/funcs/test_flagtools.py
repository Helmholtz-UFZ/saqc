#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import itertools
import operator

import numpy as np
import pandas as pd
import pytest

from saqc import BAD as B
from saqc import UNFLAGGED as U
from saqc import SaQC
from saqc.funcs.flagtools import _groupOperation
from saqc.lib.tools import toSequence
from tests.fixtures import data

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
            [1, B, -1, -1, -1],
            [N, N, B, B, N],
            {"window": 2, "method": "ffill", "dfilter": 0},
        ),
        (
            [-1, -1, -1, B, 1],
            [N, B, B, N, N],
            {"window": 2, "method": "bfill", "dfilter": 0},
        ),
        (
            [B, 1, -1, 1, 1],
            [N, N, B, N, N],
            {"window": "2D", "method": "ffill", "dfilter": 0},
        ),
        (
            [B, 1, 1, -1, 1],
            [N, N, N, B, N],
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
    result = saqc._history["x"].hist[1].astype(float)
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
    "field, target, expected, copy",
    [
        ("x", "a", [B, B, U, B], True),
        (["y", "x"], "a", [B, B, U, B], False),
        (["y", "x"], ["a", "b"], [B, B, U, B], True),
        (["y", ["x", "y"]], "a", [B, B, B, B], False),
        (["y", ["x", "y"]], ["c", ["a", "b"]], [B, B, B, B], True),
    ],
)
def test__groupOperation(field, target, expected, copy):
    base = SaQC(
        data=pd.DataFrame(
            {"x": [0, 1, 2, 3], "y": [0, 11, 22, 33], "z": [0, 111, 222, 333]}
        ),
        flags=pd.DataFrame({"x": [B, U, U, B], "y": [B, B, U, U], "z": [B, B, U, B]}),
    )
    that = SaQC(
        data=pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 11, 22, 33]}),
        flags=pd.DataFrame({"x": [U, B, U, B], "y": [U, U, B, U]}),
    )
    result = _groupOperation(
        saqc=base, field=field, target=target, func=operator.or_, group=[base, that]
    )
    targets = toSequence(itertools.chain.from_iterable(target))
    for t in targets:
        assert pd.Series(expected).equals(result.flags[t])

    # check source-target behavior
    if copy:
        fields = toSequence(itertools.chain.from_iterable(field))
        for f, t in zip(fields, targets):
            assert (result._data[f] == result._data[t]).all(axis=None)
