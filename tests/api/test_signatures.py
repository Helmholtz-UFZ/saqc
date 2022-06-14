#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import inspect
from typing import Callable, Set

import pandas as pd
import pytest

from saqc import SaQC
from saqc.core.register import FUNC_MAP, FunctionWrapper


def _filterSignature(func: Callable, skip: Set):
    sig = inspect.signature(func)
    return {k: v for k, v in sig.parameters.items() if k not in skip}


@pytest.mark.parametrize("name,func", FUNC_MAP.items())
def test_signatureConformance(name: str, func: FunctionWrapper):

    method = getattr(SaQC, name, None)
    # check a wrapper function is defined at all
    assert method is not None

    fsig = _filterSignature(func.func, {"data", "field", "flags"})
    msig = _filterSignature(method, {"self", "field"})
    assert fsig.keys() == msig.keys()

    for key, fp in fsig.items():
        mp = msig[key]
        try:
            assert fp == mp
        except AssertionError:
            assert mp.annotation == fp.annotation
            if pd.isna(fp.default) and pd.isna(mp.default):  # both NA: OK
                pass
            elif isinstance(fp.default, Callable) and isinstance(
                mp.default, Callable
            ):  # hard to check: ignore
                pass
            else:
                assert mp.default == fp.default
