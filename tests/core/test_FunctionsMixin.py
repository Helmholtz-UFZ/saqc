#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
import inspect

import pytest

from saqc.core.modules import FunctionsMixin

methods = [
    attr
    for attr in dir(FunctionsMixin)
    if callable(getattr(FunctionsMixin, attr)) and not attr.startswith("_")
]


@pytest.mark.parametrize("name", methods)
def test_redirect_call(name):
    fmixin = FunctionsMixin()
    method = getattr(fmixin, name)
    params = inspect.signature(method).parameters
    assert "field" in params
    assert "kwargs" in params
    dummy_params = dict.fromkeys(params.keys())
    dummy_params.pop("kwargs")

    err_msg = "'FunctionsMixin' object has no attribute '_wrap'"
    with pytest.raises(AttributeError, match=err_msg):
        method(**dummy_params)
