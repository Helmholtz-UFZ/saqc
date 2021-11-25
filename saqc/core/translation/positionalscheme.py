#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from saqc.core.flags import (
    Flags,
    History,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.core.translation.basescheme import (
    TranslationScheme,
    ForwardMap,
    BackwardMap,
)


class PositionalScheme(TranslationScheme):

    """
    Implements the translation from and to the flagging scheme implemented by CHS
    """

    DFILTER_DEFAULT = DOUBTFUL + 1

    _FORWARD: ForwardMap = {
        -6: UNFLAGGED,
        -5: UNFLAGGED,
        -2: UNFLAGGED,
        0: UNFLAGGED,
        1: DOUBTFUL,
        2: BAD,
    }
    _BACKWARD: BackwardMap = {
        np.nan: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def forward(self, flags: pd.DataFrame) -> Flags:
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

            # the exploded values form the History of `field`
            fflags = super()._translate(df, self._FORWARD)
            field_history = History(field_flags.index)
            for _, s in fflags.items():
                field_history.append(s)
            data[str(field)] = field_history

        return Flags(data)

    def backward(self, flags: Flags, **kwargs) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        pd.DataFrame
        """
        out = {}
        for field in flags.columns:
            thist = flags.history[field].hist.replace(self._BACKWARD).astype(int)
            # concatenate the single flag values
            ncols = thist.shape[-1]
            init = 9 * 10 ** ncols
            bases = 10 ** np.arange(ncols - 1, -1, -1)

            tflags = init + (thist * bases).sum(axis=1)
            out[field] = tflags

        return pd.DataFrame(out).fillna(-9999).astype(int)
