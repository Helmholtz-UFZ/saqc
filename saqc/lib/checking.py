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
    pd.Timedelta(freq)
    return freq


def checkDateStringSlice(s: slice):

    err = 0
    try:
        checkDateIndextStr(s.start)
        if s.stop is not None:
            checkDateIndextStr(s.stop)
    except KeyError:
        err += 1
    if err == 1:
        try:
            checkTimestampStr(s.start)
            if s.stop is not None:
                checkTimestampStr(s.stop)
        except KeyError:
            err += 1

    if err == 2:
        raise KeyError(
            f"DateStringSlice has to be defines with string key interpretable as Timestamps or DatetimIndex keys."
        )

    return s


def checkSaQC(o):
    from saqc.core import SaQC

    if isinstance(o, SaQC):
        return o
    else:
        ValueError(f"Not an SaQC object. Got {o}.")


def checkQCMembers(field, cols):
    toSeq = lambda x: [x] if isinstance(x, str) else x
    return [_v in cols for _v in toSeq(field)]


def checkFields(field, cols):
    field_exists = checkQCMembers(field, cols)
    if sum(field_exists) < len(field_exists):
        raise ValueError(f"Not all passed values are in {cols}. Got: {field}")


def checkNewFields(field, cols):
    field_exists = checkQCMembers(field, cols)
    if sum(field_exists) > 0:
        raise ValueError(f"Some Values already exist in {cols}. Got: {field}")
