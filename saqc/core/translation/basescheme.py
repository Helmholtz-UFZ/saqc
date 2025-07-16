#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import pandas as pd

from saqc import BAD, FILTER_ALL, UNFLAGGED
from saqc.core import DictOfSeries, Flags
from saqc.lib.types import EXTERNAL_FLAG

ForwardMap = Dict[EXTERNAL_FLAG, float]
BackwardMap = Dict[float, EXTERNAL_FLAG]


class TranslationScheme:  # pragma: no cover
    @property
    @abstractmethod
    def DFILTER_DEFAULT(self):
        pass

    @abstractmethod
    def __call__(self, flag: EXTERNAL_FLAG) -> float:
        pass

    @abstractmethod
    def toInternal(self, flags: pd.DataFrame | DictOfSeries) -> Flags:
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
        pass

    @abstractmethod
    def toExternal(self, flags: Flags, attrs: dict | None = None) -> DictOfSeries:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        attrs : dict or None, default None
            global meta information of saqc-object

        Returns
        -------
        pd.DataFrame
        """
        pass


class MappingScheme(TranslationScheme):
    """
    This class provides the basic translation mechanism and should serve as
    a base class for most other translation scheme.

    The general translation is realized through dictionary lookups, although
    we might need to extend this logic to also allow calls to translation
    functions in the future. Currently, at least one `dict` defining the
    'forward' translation  from 'user flags' -> 'internal flags' needs to be
    provided.
    Optionally a second `dict` can be passed to map 'internal flags' -> 'user flags',
    if the latter is not given, this 'backward' translation is inferred as
    the inverse of the 'forward' translation.

    The translation mechanism imposes a few restrictions:

    - The scheme must be well definied, i.e. we need a backward translation for
      every forward translation (each value in `self._forward` needs a key in
      `self._backward`).
    - We need translations for the special flags:
      * `saqc.constants.UNFLAGGED`
      * `saqc.constants.BAD`

    . That implies, that every valid translation scheme
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
        flags: Flags | pd.DataFrame | pd.Series | DictOfSeries,
        trans_map: ForwardMap | BackwardMap,
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
        DictOfSeries
        """
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()

        out = DictOfSeries()
        expected = pd.Index(trans_map.values())
        for field in flags.columns:
            with pd.option_context("future.no_silent_downcasting", True):
                out[field] = flags[field].replace(trans_map).infer_objects()
            diff = pd.Index(out[field]).difference(expected)
            if not diff.empty:
                raise ValueError(
                    f"following flag values could not be "
                    f"translated: {diff.drop_duplicates().to_list()}"
                )
        return out

    def __call__(self, flag: EXTERNAL_FLAG) -> float:
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
        return float(self._forward[flag])

    def toInternal(self, flags: pd.DataFrame | DictOfSeries | pd.Series) -> Flags:
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

    def toExternal(
        self,
        flags: Flags,
        attrs: dict | None = None,
    ) -> DictOfSeries:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        attrs : dict or None, default None
            global meta information of saqc-object

        Returns
        -------
        pd.DataFrame
        """
        out = self._translate(flags, self._backward)
        out.attrs = attrs or {}
        return out
