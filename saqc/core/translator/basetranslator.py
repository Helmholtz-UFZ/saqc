#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from saqc.core.lib import SaQCFunction, ColumnSelector
from typing import Dict, Optional, Union, Any, Tuple, Callable

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.core.flags import (
    Flags,
    UNFLAGGED,
    BAD,
)
from saqc.lib.types import ExternalFlag, MaterializedGraph, DiosLikeT


ForwardMap = Dict[ExternalFlag, float]
BackwardMap = Dict[float, ExternalFlag]


class Translator:
    """
    This class provides the basic translation mechanism and should serve as
    a base class for every other translation scheme.

    The general translation is realized through dictionary lookups, altough
    we might need to extend this logic to also allow calls to translation
    functions in the future. Currently at least one `dict` defining the
    'forward' translation  from 'user flags' -> 'internal flags' needs to be
    provided.
    Optionally a second `dict` can be passed to map 'internal flags' -> 'user flags',
    if the latter is not given, this 'backwards' translation will inferred as
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
    TO_MASK: Union[float, bool] = True

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
                f"need translations for the special flags `UNFLAGGED` ({UNFLAGGED}) and `BAD` ({BAD})"
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
            return (
                flag
            )  # type:  # ignore -> if flag is in `self._backward` it is of type float
        return self._forward[flag]

    @staticmethod
    def _generateInitFunction(
        flag_name: str, history: pd.Series
    ) -> Callable[[DictOfSeries, str, Flags, Any], Tuple[DictOfSeries, Flags]]:
        # NOTE:
        # Close over `flags_column` and `history_column`
        # to immitate the original function application,
        # that we cannot replicate directly because of
        # lacking information.
        # I am not entirely sure, if closing over
        # `flag_column` is really necessary or if we
        # even should close over `flags`
        def mapFlags(data: DictOfSeries, field: str, flags: Flags, **kwargs):
            flags[history.index, flag_name] = history
            return data, flags

        return mapFlags

    @staticmethod
    def buildGraph(flags: Flags) -> MaterializedGraph:
        """
        build a call graph from the external flags

        Build an `MaterializedGraph`, that can be used
        in replays of the original `SaQC` run yielding the
        same result for the same input data set.

        As we usually don't have enough information (i.e. SaQC
        function name and all used parameters) we generate dummy
        functions here. These dummy functions unconditionally set
        the `field` to the provided flags.

        Parameters
        ----------
        flags : flags to generate a call graph for
        """
        out = []
        for flag_name in flags.columns:
            # skip the default column
            for _, hist_column in tuple(flags.history[flag_name].hist.items())[1:]:
                out.append(
                    (
                        ColumnSelector(flag_name),
                        SaQCFunction(
                            name="initFlags",
                            function=Translator._generateInitFunction(
                                flag_name, hist_column
                            ),
                        ),
                    )
                )
        return out

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
        tflags = Flags(self._translate(flags, self._forward))
        return tflags, self.buildGraph(tflags)

    def backward(self, flags: Flags, call_graph: MaterializedGraph) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate
        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)
            `call_stack` is not evaluated here, it's presence only ensures, that subclasses
            have access to it.

        Returns
        -------
        pd.DataFrame
        """
        return self._translate(flags, self._backward).to_df()


class FloatTranslator(Translator):

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
