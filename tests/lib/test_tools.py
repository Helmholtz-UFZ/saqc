# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

import saqc.lib.tools as tools
from saqc.lib.checking import validateScalar


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("value", [1, 0, "foo", np.nan, np.inf])
def test_validateScalar(value, optional):
    validateScalar(value, "value", optional)


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("value", [[1], [0, 1], {}, {1, 2}, pd.Series([1, 2])])
def test_validateScalar(value, optional):
    with pytest.raises(ValueError):
        validateScalar(value, "value", optional)


def test_validateScalar():
    validateScalar("value", None, optional=True)
    with pytest.raises(ValueError):
        validateScalar(None, "value", optional=False)


class _ListLike(list):
    pass


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, [None]),
        ([None], [None]),
        (1, [1]),
        (np.nan, [np.nan]),
        ([1], [1]),
        ("foo", ["foo"]),
        (["foo"], ["foo"]),
        ([1, 2], [1, 2]),
        (_ListLike("ab"), ["a", "b"]),
    ],
)
def test_toSequence(value, expected):
    result = tools.toSequence(value)
    assert isinstance(result, list)
    assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        # squeeze
        ([1], 1),
        ([[1, 2]], [1, 2]),
        ([[]], []),
        (_ListLike("a"), "a"),
        # no squeeze
        ([], []),
        ([1, 2], [1, 2]),
        (_ListLike("ab"), ["a", "b"]),
    ],
)
def test_squeezeSequence(value, expected):
    result = tools.squeezeSequence(value)
    assert result == expected
