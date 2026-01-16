#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

import pandas as pd

T = TypeVar("T")


def checkTimestampStr(s: str) -> pd.offsets.BaseOffset:
    try:
        pd.Timestamp(s)
    except:
        raise KeyError(f"can't associate '{s}' with a timestamp.")
    return s


def checkPathStr(s: str) -> pd.offsets.BaseOffset:
    try:
        Path(s)
    except TypeError:
        raise KeyError(f"Cant make no path from: '{s}'")
    return s


def checkDateIndextStr(s: str) -> pd.offsets.BaseOffset:
    try:
        pd.Series(index=pd.DatetimeIndex([]))[s]
    except KeyError:
        raise KeyError(f"can't index with: '{s}'")
    return s


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


def checkQCMembers(field, cols, is_regex):
    toSeq = lambda x: [x] if isinstance(x, str) else x
    field = toSeq(field)
    if not is_regex:
        field_exists = [_v in cols for _v in toSeq(field)]
    else:
        field_exists = [sum([re.match(field[0], _v) is not None for _v in cols]) > 0]
    return field_exists


def checkFields(field, cols, is_regex=False):
    field_exists = checkQCMembers(field, cols, is_regex)
    if sum(field_exists) < len(field_exists):
        raise ValueError(f"Not all passed values are in {cols}. Got: {field}")


def checkNewFields(field, cols):
    field_exists = checkQCMembers(field, cols, False)
    if sum(field_exists) > 0:
        raise ValueError(f"Some Values already exist in {cols}. Got: {field}")
