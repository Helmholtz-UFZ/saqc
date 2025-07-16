#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TypeVar

import pandas as pd

T = TypeVar("T")


#
def checkOffsetStr(freq: str) -> pd.offsets.BaseOffset:
    try:
        pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        raise ValueError(f"Not an offset reference: '{freq}'")
    return freq


def checkFreqStr(freq: str) -> pd.offsets.BaseOffset:
    try:
        f = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        raise ValueError(f"Not an offset reference: '{freq}'")
    try:
        pd.Timedelta(f)
    except ValueError:
        raise ValueError(
            f"Not a frequency string: {freq}. \n "
            f"-> {freq} refers to an Offset (={f}). But that cant be interpreted as Frequency (most likely because its not a fixed temporal extension)."
        )
    return freq


def checkSaQC(o):
    from saqc.core import SaQC

    if isinstance(o, SaQC):
        return o
    else:
        ValueError(f"Not an SaQC object. Got {o}.")


def checkQCMembers(field, cols):
    toSeq = lambda x: [x] if isinstance(x, str) else x
    field = toSeq(field)
    field_exists = [_v in cols for _v in toSeq(field)]
    return field_exists


def checkFields(field, cols):
    field_exists = checkQCMembers(field, cols)
    if sum(field_exists) < len(field_exists):
        raise ValueError(f"Not all passed values are in {cols}. Got: {field}")


def checkNewFields(field, cols):
    field_exists = checkQCMembers(field, cols)
    if sum(field_exists) > 0:
        raise ValueError(f"Some Values already exist in {cols}. Got: {field}")
