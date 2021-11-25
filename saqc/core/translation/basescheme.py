#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Union, Any

import numpy as np
import pandas as pd

from dios import DictOfSeries
from saqc.constants import (
    FILTER_ALL,
    UNFLAGGED,
    BAD,
    GOOD,
)

from saqc.core.flags import Flags
from saqc.lib.types import ExternalFlag


ForwardMap = Dict[ExternalFlag, float]
BackwardMap = Dict[float, ExternalFlag]


class TranslationScheme:
    """
    This class provides the basic translation mechanism and should serve as
    a base class for every other translation scheme.

    The general translation is realized through dictionary lookups, altough
    we might need to extend this logic to also allow calls to translation
    functions in the future. Currently at least one `dict` defining the
    'forward' translation  from 'user flags' -> 'internal flags' needs to be
    provided.
    Optionally a second `dict` can be passed to map 'internal flags' -> 'user flags',
    if the latter is not given, this 'backward' translation will inferred as
    the inverse of the 'forward' translation.

    The translation mechanism imposes a few restrictions:
    - The scheme must be well definied, i.e. we need a backward translation for
      every forward translation (each value in `self._forward` needs a key in
      `self._backward`).
    - We need translations for the special flags `saqc.constants.UNFLAGGED` and
      `saqc.constants.BAD`. That implies, that every valid translation scheme
      provides at least one user flag that maps to `BAD` and one that maps to
      `UNFLAGGED`.
    """

    # (internal) threshold flag above which values will be masked
    DFILTER_DEFAULT: float = FILTER_ALL

    # additional arguments and default values the translation scheme accepts
    ARGUMENTS: Dict[str, Any] = {}

    def __init__(self, forward: ForwardMap, backward: BackwardMap):
        """
        Parameters
        ----------
        forward : dict
            A mapping defining the forward translation of scalar flag values

        backward : dict
            A mapping defining the backward translation of scalar flag values

        Note
        ----
        `backward` needs to provide a mapping for the two special flags
        `saqc.constants.UNFLAGGED`, `saqc.constants.BAD`
        """
        if UNFLAGGED not in backward or BAD not in backward:
            raise ValueError(
                f"need translations for the special flags `UNFLAGGED` ({UNFLAGGED})"
                f" and `BAD` ({BAD})"
            )
        self._forward = forward
        self._backward = backward

    @staticmethod
    def _translate(
        flags: Union[Flags, pd.DataFrame, pd.Series],
        trans_map: Union[ForwardMap, BackwardMap],
    ) -> DictOfSeries:
        """
        Translate a given flag data structure to another according to the
        mapping given in `trans_map`

        Parameters
        ----------
        flags : Flags, pd.DataFrame
            The flags to translate

        Returns
        -------
        pd.DataFrame, Flags
        """
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()

        out = DictOfSeries()
        expected = pd.Index(trans_map.values())
        for field in flags.columns:
            out[field] = flags[field].replace(trans_map)
            diff = pd.Index(out[field]).difference(expected)
            if not diff.empty:
                raise ValueError(
                    f"flags were not translated: {diff.drop_duplicates().to_list()}"
                )
        return out

    def __call__(self, flag: ExternalFlag) -> float:
        """
        Translate a scalar 'external flag' to an 'internal flag'

        Parameters
        ----------
        flag : float, int, str
            The external flag to translate

        Returns
        -------
        float
        """
        if flag not in self._forward:
            if flag not in self._backward:
                raise ValueError(f"invalid flag: {flag}")
            return float(flag)
        return self._forward[flag]

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
        return Flags(self._translate(flags, self._forward))

    def backward(
        self,
        flags: Flags,
        raw: bool = False,
        attrs: dict | None = None,
        **kwargs,
    ) -> Union[pd.DataFrame, DictOfSeries]:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        raw: bool, default False
            if True return data as DictOfSeries, otherwise as pandas DataFrame.

        attrs : dict or None, default None
            global meta information of saqc-object

        Returns
        -------
        pd.DataFrame
        """
        out = self._translate(flags, self._backward)
        out.attrs = attrs or {}
        if not raw:
            out = out.to_df()
        return out


class FloatScheme(TranslationScheme):

    """
    Acts as the default Translator, provides a changeable subset of the
    internal float flags
    """

    _MAP = {
        -np.inf: -np.inf,
        **{k: k for k in np.arange(0, 256, dtype=float)},
    }

    def __init__(self):
        super().__init__(self._MAP, self._MAP)


class SimpleScheme(TranslationScheme):

    """
    Acts as the default Translator, provides a changeable subset of the
    internal float flags
    """

    _FORWARD = {
        "UNFLAGGED": UNFLAGGED,
        "BAD": BAD,
        "OK": GOOD,
    }

    _BACKWARD = {
        UNFLAGGED: "UNFLAGGED",
        np.nan: "UNFLAGGED",
        BAD: "BAD",
        GOOD: "OK",
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)
