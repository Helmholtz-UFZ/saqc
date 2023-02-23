#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pytest

from saqc import BAD, UNFLAGGED
from saqc.core import SaQC, initFlagsLike
from tests.common import initData


@pytest.fixture
def data():
    constants_data = initData(
        1, start_date="2011-01-01 00:00:00", end_date="2011-01-01 03:00:00", freq="5min"
    )
    for c in constants_data.columns:
        constants_data[c].iloc[5:25] = 200
    return constants_data


def test_constants_flagBasic(data):
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags)
    qc = qc.flagConstants(field, window="15Min", thresh=0.1, flag=BAD)
    flagscol = qc._flags[field]
    assert np.all(flagscol[5:25] == BAD)
    assert np.all(flagscol[:5] == UNFLAGGED)
    assert np.all(flagscol[25 + 1 :] == UNFLAGGED)


def test_constants_flagVarianceBased(data):
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags)
    qc = qc.flagByVariance(field, window="1h", thresh=0.0005, flag=BAD)

    flagscol = qc._flags[field]
    assert np.all(flagscol[5:25] == BAD)
    assert np.all(flagscol[:5] == UNFLAGGED)
    assert np.all(flagscol[25 + 1 :] == UNFLAGGED)
