#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from saqc import BAD, DOUBTFUL, GOOD, UNFLAGGED
from saqc.core import Flags, History
from saqc.core.frame import DictOfSeries
from saqc.core.translation.basescheme import BackwardMap, ForwardMap, MappingScheme

_QUALITY_CAUSES = [
    "",
    "BATTERY_LOW",
    "BELOW_MINIMUM",
    "ABOVE_MAXIMUM",
    "BELOW_OR_ABOVE_MIN_MAX",
    "ISOLATED_SPIKE",
    "DEFECTIVE_SENSOR",
    "LEFT_CENSORED_DATA",
    "RIGHT_CENSORED_DATA",
    "OTHER",
    "AUTOFLAGGED",
]

_QUALITY_LABELS = [
    "quality_flag",
    "quality_cause",
    "quality_comment",
]


class DmpScheme(MappingScheme):
    """
    Implements the translation from and to the flagging scheme implemented in
    the UFZ - Datamanagementportal
    """

    ARGUMENTS = {"comment": "", "cause": "OTHER"}

    DFILTER_DEFAULT = GOOD + 1

    _FORWARD: ForwardMap = {
        "NIL": UNFLAGGED,
        "OK": GOOD,
        "DOUBTFUL": DOUBTFUL,
        "BAD": BAD,
    }

    _BACKWARD: BackwardMap = {
        UNFLAGGED: "NIL",
        np.nan: "NIL",
        GOOD: "OK",
        DOUBTFUL: "DOUBTFUL",
        BAD: "BAD",
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def toHistory(self, flags: pd.DataFrame):
        """
        Translate a single field of external ``Flags`` to a ``History``
        """
        history = History(flags.index)

        for (flag, cause, comment), values in flags.groupby(_QUALITY_LABELS):
            if cause == "" and comment == "":
                continue

            try:
                comment = json.loads(comment)
            except json.decoder.JSONDecodeError:
                comment = {"test": "unknown", "comment": ""}

            column = pd.Series(np.nan, index=flags.index)
            column.loc[values.index] = self(flag)

            meta = {
                "func": comment["test"],
                "kwargs": {"comment": comment["comment"], "cause": cause},
            }
            history.append(column, meta=meta)
        return history

    def toInternal(self, flags: pd.DataFrame | DictOfSeries) -> Flags:
        """
        Translate from 'external flags' to 'internal flags'

        Parameters
        ----------
        df : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """

        if isinstance(flags, pd.DataFrame):
            flags = DictOfSeries(flags)

        self.validityCheck(flags)

        data = {}

        if isinstance(flags, pd.DataFrame):
            fields = flags.columns.get_level_values(0).drop_duplicates()
        else:
            fields = flags.keys()

        for field in fields:
            data[str(field)] = self.toHistory(flags[field])

        return Flags(data)

    def toExternal(
        self, flags: Flags, attrs: dict | None = None, **kwargs
    ) -> DictOfSeries:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : The external flags to translate

        attrs : dict or None, default None
            global meta information of saqc-object

        Returns
        -------
        translated flags
        """
        tflags = super().toExternal(flags, attrs=attrs)

        out = DictOfSeries()

        for field in tflags.columns:
            df = pd.DataFrame(
                {
                    "quality_flag": tflags[field],
                    "quality_cause": "",
                    "quality_comment": "",
                }
            )

            history = flags.history[field]
            for col in history.columns:
                valid = (history.hist[col] != UNFLAGGED) & history.hist[col].notna()

                # extract from meta
                meta = history.meta[col]
                keywords = meta.get("kwargs", {})
                comment = json.dumps(
                    {
                        "test": meta.get("func", "unknown"),
                        "comment": keywords.get("comment", self.ARGUMENTS["comment"]),
                    }
                )
                cause = keywords.get("cause", self.ARGUMENTS["cause"])
                df.loc[valid, "quality_comment"] = comment
                df.loc[valid, "quality_cause"] = cause

            out[field] = df

        self.validityCheck(out)
        return out

    @classmethod
    def validityCheck(cls, flags: DictOfSeries) -> None:
        """
        Check wether the given causes and comments are valid.

        Parameters
        ----------
        df : external flags
        """
        for df in flags.values():

            if not df.columns.isin(_QUALITY_LABELS).all(axis=None):
                raise TypeError(
                    f"DMP-Flags expect the labels {list(_QUALITY_LABELS)} in the secondary level"
                )

            flags = df["quality_flag"]
            causes = df["quality_cause"]
            comments = df["quality_comment"]

            if not flags.isin(cls._FORWARD.keys()).all(axis=None):
                raise ValueError(
                    f"invalid quality flag(s) found, only the following values are supported: {set(cls._FORWARD.keys())}"
                )

            if not causes.isin(_QUALITY_CAUSES).all(axis=None):
                raise ValueError(
                    f"invalid quality cause(s) found, only the following values are supported: {_QUALITY_CAUSES}"
                )

            if (~flags.isin(("OK", "NIL")) & (causes == "")).any(axis=None):
                raise ValueError(
                    "quality flags other than 'OK and 'NIL' need a non-empty quality cause"
                )

            if ((causes == "OTHER") & (comments == "")).any(axis=None):
                raise ValueError(
                    "quality cause 'OTHER' needs a non-empty quality comment"
                )
