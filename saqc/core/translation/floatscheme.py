#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from saqc.constants import FILTER_ALL, UNFLAGGED
from saqc.core.flags import Flags
from saqc.core.frame import DictOfSeries
from saqc.core.history import History
from saqc.core.translation.basescheme import TranslationScheme


class FloatScheme(TranslationScheme):
    """
    Acts as the default Translator, provides a changeable subset of the
    internal float flags
    """

    DFILTER_DEFAULT: float = FILTER_ALL

    def __call__(self, flag: float | int) -> float:
        try:
            return float(flag)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"invalid flag, expected a numerical value, got: {flag}")

    def toInternal(self, flags: pd.DataFrame | DictOfSeries) -> Flags:
        try:
            return Flags(flags.astype(float))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"invalid flag(s), expected a collection of numerical values, got: {flags}"
            )

    def toExternal(self, flags: Flags, attrs: dict | None = None) -> DictOfSeries:
        out = DictOfSeries(flags)
        out.attrs = attrs or {}
        return out


class AnnotatedFloatScheme(FloatScheme):
    def toExternal(self, flags: Flags, attrs: dict | None = None) -> DictOfSeries:

        tflags = super().toExternal(flags, attrs=attrs)

        out = DictOfSeries()
        for field in tflags.columns:
            df = pd.DataFrame(
                {
                    "flag": tflags[field],
                    "func": "",
                    "parameters": "",
                }
            )

            history = flags.history[field]

            for col in history.columns:
                valid = (history.hist[col] != UNFLAGGED) & history.hist[col].notna()
                meta = history.meta[col]
                df.loc[valid, "func"] = meta["func"]
                df.loc[valid, "parameters"] = str(meta["kwargs"])
                out[field] = df

        return out

    def toInternal(self, flags: DictOfSeries) -> Flags:
        data = {}
        for key, frame in flags.items():
            history = History(index=frame.index)
            for (flag, func, kwargs), values in frame.groupby(
                ["flag", "func", "parameters"]
            ):
                column = pd.Series(np.nan, index=frame.index)
                column.loc[values.index] = self(flag)
                history.append(column, meta={"func": func, "kwargs": kwargs})
            data[key] = history
        return Flags(data)
