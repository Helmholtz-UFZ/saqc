#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*field-target flagging workflows is deprecated.*",
        category=FutureWarning,
    )
    import saqc


@pytest.fixture(autouse=True)
def field_target_append():
    saqc.options.field_target.existing_target = "append"
    yield
    saqc.options.field_target.existing_target = "append"
