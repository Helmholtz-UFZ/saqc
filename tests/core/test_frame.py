#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-


import pytest

from saqc.core.frame import DictOfSeries as DoS
from saqc.core.frame import concatDios


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
    result = concatDios(data, warn=False)
    assert result == expected


@pytest.mark.parametrize("data", [[DoS(dict(a=[1], b=[2])), DoS(dict(b=[22]))]])
def test_concatDios_warning(data):
    with pytest.warns(UserWarning):
        concatDios(data, warn=True, stacklevel=0)
