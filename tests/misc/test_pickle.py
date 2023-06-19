#!/usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pickle

import pytest

from saqc import SaQC
from tests.common import checkInvariants, initData


@pytest.mark.parametrize("ncols", [4, 0])
def test_pickling(ncols):
    """Ensure that saqc and all its parts are pickleable"""
    qc = SaQC(data=initData(ncols))
    result = pickle.loads(pickle.dumps(qc))
    assert isinstance(result, SaQC)
    for k in qc.data.keys():
        checkInvariants(qc._data, qc._flags, k)
