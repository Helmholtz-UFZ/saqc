# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import saqc.lib.ts_operators as tsops


def test_butterFilter():
    assert (
        tsops.butterFilter(pd.Series([1, -1] * 100), cutoff=0.1)
        - pd.Series([1, -1] * 100)
    ).mean() < 0.5


T = True
F = False


@pytest.mark.parametrize(
    "arr,maxc,expected",
    [
        (np.array([]), 1, False),
        (np.array([F]), 1, False),
        (np.array([F, F, F]), 1, False),
        #
        (np.array([T]), 0, True),
        (np.array([T]), 1, False),
        #
        (np.array([F, T, F]), 0, True),
        (np.array([F, T, F]), 1, False),
        #
        (np.array([F, T, T, T, T, F]), 0, True),
        (np.array([F, T, T, T, T, F]), 1, True),
        (np.array([F, T, T, T, T, F]), 2, True),
        (np.array([F, T, T, T, T, F]), 3, True),
        (np.array([F, T, T, T, T, F]), 4, False),
        (np.array([F, T, T, T, T, F]), 5, False),
        #
        (np.array([F, T, T, F, T, T, F]), 2, False),
    ],
)
def test__exceedConsecutiveNanLimit(arr, maxc, expected):
    result = tsops._exceedConsecutiveNanLimit(arr, maxc)
    assert result is expected


def dtSeries(data, freq="1d"):
    index = pd.date_range(start="2020", periods=len(data), freq=freq)
    return pd.Series(data=data, index=index, dtype=float)


@pytest.mark.parametrize(
    "data",
    [dtSeries([0, 1, 2]), dtSeries([0, np.nan, 2])],
)
def test_identity(data):
    from saqc.lib.ts_operators import identity

    result = identity(data)
    assert result is data


@pytest.mark.parametrize(
    "data,expected",
    [
        (dtSeries([0, 1, 2]), 3),
        (dtSeries([0, np.nan, 2]), 2),
    ],
)
def test_count(data, expected):
    # count is labeled as a dummy function, this means
    # we need to ensure it exists with a resampler object.
    resampler = data.resample("2d")
    assert hasattr(resampler, "count")

    from saqc.lib.ts_operators import count

    result = count(data)
    assert result == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            dtSeries([1, 2, np.inf, np.nan]),
            dtSeries([np.log(1), np.log(2), np.inf, np.nan]),
        ),
        pytest.param(
            dtSeries(
                [
                    0,
                    -2,
                    -1,
                    -np.inf,
                ]
            ),
            dtSeries([np.nan, np.nan, np.nan, np.nan]),
            marks=pytest.mark.xfail(reason="zeroLog(0) did not return NaN for 0"),
        ),
    ],
)
def test_zeroLog(data, expected):
    from saqc.lib.ts_operators import zeroLog

    result = zeroLog(data)
    assert_series_equal(result, expected, check_freq=False, check_names=False)


@pytest.mark.parametrize(
    "data,expected",
    [
        (dtSeries([1, 2, 3]), dtSeries([np.nan, 1440, 1440])),
        (
            pd.Series(
                [1, 2, 3],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-13"]),
            ),
            dtSeries([np.nan, 2880, 14400]),
        ),
    ],
)
def test_deltaT(data, expected):
    from saqc.lib.ts_operators import deltaT

    result = deltaT(data)
    assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_names=False,
        check_index=False,
        check_freq=False,
    )


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param(
            pd.Series(
                # We use as values the delta of total seconds from the last value.
                # This way the 'derivative' should be 1 for each result value.
                [1, 2880, 14400],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-13"]),
            ),
            pd.Series(
                [np.nan, 1, 1],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-13"]),
            ),
        ),
    ],
)
def test_derivative(data, expected):
    from saqc.lib.ts_operators import derivative

    result = derivative(data)
    assert_series_equal(result, expected, check_dtype=False, check_names=False)


@pytest.mark.parametrize(
    "data,expected",
    [
        (dtSeries([1, 1, 1]), dtSeries([np.nan, 0, 0])),
        (dtSeries([1, 10, 100]), dtSeries([np.nan, 9, 90])),
        (dtSeries([-np.inf, np.inf, 0]), dtSeries([np.nan, np.inf, -np.inf])),
        (dtSeries([0, np.nan, 0]), dtSeries([np.nan, np.nan, np.nan])),
    ],
)
def test_difference(data, expected):
    from saqc.lib.ts_operators import difference

    result = difference(data)
    assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    "data,expected",
    [
        (dtSeries([1, 1, 1]), dtSeries([np.nan, 0, 0])),
        (dtSeries([1, 10, 100]), dtSeries([np.nan, 0.9, 0.9])),
        (dtSeries([-np.inf, np.inf, 0]), dtSeries([np.nan, np.nan, -np.inf])),
        (dtSeries([0, np.nan, 0]), dtSeries([np.nan, np.nan, np.nan])),
    ],
)
def test_rateOfChange(data, expected):
    from saqc.lib.ts_operators import rateOfChange

    result = rateOfChange(data)
    assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    "limit,extrapolate,data,expected",
    [
        (
            1,
            None,
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
        ),
        (
            2,
            "backward",
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [0, 0, np.nan, np.nan, np.nan, 4, np.nan],
        ),
        (
            2,
            None,
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
        ),
        (
            3,
            None,
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
        ),
        (
            3,
            "forward",
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, np.nan, np.nan, np.nan, 4, 4],
        ),
        (
            4,
            None,
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, 1, 2, 3, 4, np.nan],
        ),
        (
            4,
            "both",
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, 1, 2, 3, 4, np.nan],
        ),
        (
            None,
            None,
            [np.nan, 0, np.nan, np.nan, np.nan, 4, np.nan],
            [np.nan, 0, 1, 2, 3, 4, np.nan],
        ),
    ],
)
def test_interpolatNANs(limit, extrapolate, data, expected):
    got = tsops.interpolateNANs(
        pd.Series(data), gap_limit=limit, method="linear", extrapolate=extrapolate
    )
    try:
        assert got.equals(pd.Series(expected, dtype=float))
    except AssertionError:
        print("stop")
