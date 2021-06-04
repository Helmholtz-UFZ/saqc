#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import pandas as pd

from saqc.core.flags import (
    Flags,
    _simpleHist,
    UNTOUCHED,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.lib.types import MaterializedGraph
from saqc.core.translator.basetranslator import Translator, ForwardMap, BackwardMap


class PositionalTranslator(Translator):

    """
    Implements the translation from and to the flagging scheme implemented by CHS
    """

    _FORWARD: ForwardMap = {0: UNFLAGGED, 1: DOUBTFUL, 2: BAD}
    _BACKWARD: BackwardMap = {
        UNTOUCHED: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def forward(self, flags: pd.DataFrame) -> Tuple[Flags, MaterializedGraph]:
        """
        Translate from 'external flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """

        data = {}
        for field, field_flags in flags.items():

            # explode the flags into sperate columns and drop the leading `9`
            df = pd.DataFrame(
                field_flags.astype(str).str.slice(start=1).apply(tuple).tolist(),
                index=field_flags.index,
            ).astype(int)

            # the exploded values + the an initial column are the History of `field`
            fflags = super()._translate(df, self._FORWARD)
            field_history = _simpleHist(field_flags.index).append(fflags.to_df())
            data[field] = field_history

        tflags = Flags(data)
        graph = self.buildGraph(tflags)
        return tflags, graph

    def backward(self, flags: Flags, call_stack: MaterializedGraph) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate
        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)
            `call_stack` is not evaluated here.

        Returns
        -------
        pd.DataFrame
        """
        out = {}
        for field in flags.columns:
            thist = flags.history[field].hist.replace(self._BACKWARD)
            # Concatenate the single flag values. There are faster and more
            # complicated approaches (see former `PositionalFlagger`), but
            # this method shouldn't be called that often
            tflags = (
                thist.astype(int).astype(str).apply(lambda x: x.sum(), axis="columns")
            )
            out[field] = "9"
            if not tflags.empty:
                # take care for the default columns
                out[field] += tflags.str.slice(start=1)

        return pd.DataFrame(out).fillna(-9999).astype(int)
