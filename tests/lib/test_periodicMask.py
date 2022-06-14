# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from saqc.lib.tools import periodicMask

unmasked, masked = True, False


def assert_equal(result, exp):
    assert exp.index.equals(result.index)
    assert result.dtype == exp.dtype

    if result.equals(exp):
        return

    df = pd.DataFrame(
        data=np.array([result, exp]).T, index=exp.index, columns=["result", "exp"]
    )
    print("\n", df)
    assert result.equals(exp)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(
            [
                "1999-10",
                "1999-11",
                "1999-12",
                "2000-01",
                "2000-02",
                "2000-03",
            ]
        )
    ],
)
@pytest.mark.parametrize(
    "start, end, closed, exp",
    [
        (
            "01-01T00:00:00",
            "03-01T00:00:00",
            True,
            [unmasked, unmasked, unmasked, masked, masked, masked],
        ),
        (
            "01-01T00:00:00",
            "03-01T00:00:00",
            False,
            [unmasked, unmasked, unmasked, unmasked, masked, unmasked],
        ),
        (
            "03-01T00:00:00",
            "01-01T00:00:00",
            True,
            [masked, masked, masked, masked, unmasked, masked],
        ),
        (
            "03-01T00:00:00",
            "01-01T00:00:00",
            False,
            [masked, masked, masked, unmasked, unmasked, unmasked],
        ),
    ],
)
def test_bounds(index, start, end, closed, exp):
    exp = pd.Series(exp, index=index, dtype=bool)
    result = periodicMask(index, start, end, closed)
    assert_equal(result, exp)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(
            [
                "1990-01",
                "1990-02",
                "1990-03",
                "1990-04",
                "2000-01",
                "2000-02",
                "2000-03",
                "2000-04",
            ]
        )
    ],
)
@pytest.mark.parametrize(
    "start, end, closed, exp",
    [
        (
            "01-01T00:00:00",
            "03-01T00:00:00",
            True,
            [masked, masked, masked, unmasked, masked, masked, masked, unmasked],
        ),
        (
            "01-01T00:00:00",
            "03-01T00:00:00",
            False,
            [
                unmasked,
                masked,
                unmasked,
                unmasked,
                unmasked,
                masked,
                unmasked,
                unmasked,
            ],
        ),
    ],
)
def test_season(index, start, end, closed, exp):
    exp = pd.Series(exp, index=index, dtype=bool)
    result = periodicMask(index, start, end, closed)
    assert_equal(result, exp)
