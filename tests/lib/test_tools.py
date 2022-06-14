# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

import saqc.lib.tools as tools
from dios import DictOfSeries as DoS


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("value", [1, 0, "foo", np.nan, np.inf])
def test_assertScalar(value, optional):
    tools.assertScalar("value", value, optional)


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("value", [[1], [0, 1], {}, {1, 2}, pd.Series([1, 2])])
def test_assertScalar_raises(value, optional):
    with pytest.raises(ValueError):
        tools.assertScalar("value", value, optional)


def test_assertScalar_optional():
    tools.assertScalar("value", None, optional=True)
    with pytest.raises(ValueError):
        tools.assertScalar("value", None, optional=False)


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


@pytest.mark.parametrize(
    "data, expected",
    [
        # 2c + 1c -> 3c
        ([DoS(dict(a=[1], b=[2])), DoS(dict(c=[3]))], DoS(dict(a=[1], b=[2], c=[3]))),
        # 1c + 1c + 1c -> 3c
        (
            [DoS(dict(a=[1])), DoS(dict(b=[1])), DoS(dict(c=[1]))],
            DoS(dict(a=[1], b=[1], c=[1])),
        ),
        # 2c + 1c (overwrite) = 2c
        ([DoS(dict(a=[1], b=[2])), DoS(dict(b=[22]))], DoS(dict(a=[1], b=[22]))),
        # 1c + 1c + 1c (all overwrite) -> 1c
        (
            [DoS(dict(a=[1])), DoS(dict(a=[11])), DoS(dict(a=[111]))],
            DoS(dict(a=[111])),
        ),
    ],
)
def test_concatDios(data, expected):
    result = tools.concatDios(data, warn=False)
    assert result == expected


@pytest.mark.parametrize("data", [[DoS(dict(a=[1], b=[2])), DoS(dict(b=[22]))]])
def test_concatDios_warning(data):
    with pytest.warns(UserWarning):
        tools.concatDios(data, warn=True, stacklevel=0)
