#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
import pytest

from saqc import BAD as B
from saqc import UNFLAGGED as U
from saqc import SaQC

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
