#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Any
from functools import reduce

import numpy as np
import pandas as pd

from saqc.core.history import History
from saqc.core.flags import (
    Flags,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.core.translation.basescheme import (
    BackwardMap,
    TranslationScheme,
    ForwardMap,
)


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


class DmpScheme(TranslationScheme):

    """
    Implements the translation from and to the flagging scheme implemented in
    the UFZ - Datamanagementportal
    """

    ARGUMENTS = {"comment": "", "cause": "OTHER"}

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

    def forward(self, df: pd.DataFrame) -> Flags:
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

        self.validityCheck(df)

        data = {}

        for field in df.columns.get_level_values(0):

            field_flags = df[field]
            field_history = History(field_flags.index)

            for (flag, cause, comment), values in field_flags.groupby(_QUALITY_LABELS):
                try:
                    comment = json.loads(comment)
                except json.decoder.JSONDecodeError:
                    comment = {"test": "unknown", "comment": ""}

                histcol = pd.Series(np.nan, index=field_flags.index)
                histcol.loc[values.index] = self(flag)

                meta = {
                    "func": comment["test"],
                    "kwargs": {"comment": comment["comment"], "cause": cause},
                }
                field_history.append(histcol, meta=meta)

            data[str(field)] = field_history

        return Flags(data)

    def backward(
        self, flags: Flags, attrs: dict | None = None, **kwargs
    ) -> pd.DataFrame:
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
        tflags = super().backward(flags, raw=True, attrs=attrs)

        out = pd.DataFrame(
            index=reduce(lambda x, y: x.union(y), tflags.indexes).sort_values(),
            columns=pd.MultiIndex.from_product([flags.columns, _QUALITY_LABELS]),
        )

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

            out[field] = df.reindex(out.index)

        self.validityCheck(out)
        return out

    @classmethod
    def validityCheck(cls, df: pd.DataFrame) -> None:
        """
        Check wether the given causes and comments are valid.

        Parameters
        ----------
        df : external flags
        """

        cols = df.columns
        if not isinstance(cols, pd.MultiIndex):
            raise TypeError("DMP-Flags need multi-index columns")

        if not cols.get_level_values(1).isin(_QUALITY_LABELS).all(axis=None):
            raise TypeError(
                f"DMP-Flags expect the labels {list(_QUALITY_LABELS)} in the secondary level"
            )

        for field in df.columns.get_level_values(0):

            # we might have NaN injected by DictOfSeries -> DataFrame conversions
            field_df = df[field].dropna(how="all", axis="index")
            flags = field_df["quality_flag"]
            causes = field_df["quality_cause"]
            comments = field_df["quality_comment"]

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

            if ((causes == "OTHER") & (comments == "")).any(None):
                raise ValueError(
                    "quality cause 'OTHER' needs a non-empty quality comment"
                )
