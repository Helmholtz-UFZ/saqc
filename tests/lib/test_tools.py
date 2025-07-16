# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

import saqc.lib.tools as tools


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
