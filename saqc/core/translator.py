#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.core.flags import Flags, UNTOUCHED, UNFLAGGED, GOOD, DOUBTFUL, BAD
from saqc.core.history import History
from saqc.lib.types import ExternalFlag, MaterializedGraph

# to_mask as part of th translator

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

    TO_MASK = True

    def __init__(self, forward: ForwardMap, backward: Optional[BackwardMap] = None):
        """
        Parameters
        ----------
        forward : dict
            A mapping defining the forward translation of scalar flag values

        backward : dict, optinal
            A mapping defining the backward translation of scalar flag values.
            If not given, `backward` is inferred from `forward`

        Note
        ----
        `backward` needs to provide a mappinf for the two special flags
        `saqc.core.UNFLAGGED`, `saqc.core.BAD`
        """
        # NOTE: we also add the keys to also allow the usage of internal flags
        self._forward = forward
        if backward is None:
            backward = {v: k for k, v in forward.items()}
        if {UNFLAGGED, BAD} - set(backward.keys()):
            raise ValueError(
                f"need translations for the special flags `UNFLAGGED` ({UNFLAGGED}) and `BAD` ({BAD})"
            )
        self._backward = backward

    @staticmethod
    def _translate(
        flags: Union[Flags, pd.DataFrame], trans_map: Union[ForwardMap, BackwardMap]
    ) -> DictOfSeries:
        """
        Translate a given flag data structure to another one according to the
        mapping given in `trans_map`

        Parameters
        ----------
        flags : Flags, pd.DataFrame
            The flags to translate

        Returns
        -------
        pd.DataFrame, Flags
        """
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
            return flag  # type: ignore  -> if flag is in `self._backward` it is of type float
        return self._forward[flag]

    def forward(self, flags: pd.DataFrame) -> Flags:
        """
        Translate from 'extrnal flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """
        return Flags(self._translate(flags, self._forward))

    def backward(self, flags: Flags, call_stack: MaterializedGraph) -> pd.DataFrame:
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
        # NOTE:
        return self._translate(flags, self._backward).to_df()


class FloatTranslator(Translator):

    """
    Acts as the default Translator, provides a changeable subset of the
    internal float flags
    """

    _FORWARD: Dict[float, float] = {
        -np.inf: -np.inf,
        **{k: k for k in np.arange(0, 256, dtype=float)},
    }

    def __init__(self):
        super().__init__(self._FORWARD)


class DmpTranslator(Translator):

    """
    Implements the translation from and to the flagging scheme implemented in
    the UFZ - Datamanagementportal
    """

    _FORWARD: Dict[str, float] = {
        "NIL": UNFLAGGED,
        "OK": GOOD,
        "DOUBTFUL": DOUBTFUL,
        "BAD": BAD,
    }
    _COL_LABELS: Dict[str, str] = {
        "flag": "quality_flag",
        "comment": "quality_comment",
        "cause": "quality_cause",
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD)

    @staticmethod
    def _getFieldFunctions(field: str, call_stack: MaterializedGraph) -> List[str]:
        """
        Return the names of all functions called on `field`

        Parameters
        ----------
        field: str
            variable/column name

        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)

        Note
        ----
        Could (and maybe should) be implemented as a method of `CalledStack`
        """
        return [f.name for l, f in call_stack if l.field == field]

    def forward(self, flags: pd.DataFrame) -> Flags:
        """
        Translate from 'extrnal flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """
        cols = flags.columns
        if not isinstance(cols, pd.MultiIndex):
            raise TypeError("DMP-Flags need mult-index columns")
        if set(cols.get_level_values(1)) != set(self._COL_LABELS.values()):
            raise TypeError(
                f"DMP-Flags expect the labels 'list(self._COL_LABELS.values)' in the secondary level"
            )

        qflags = flags.xs(key=self._COL_LABELS["flag"], axis="columns", level=1)
        return super().forward(qflags)  # type: ignore

    def backward(self, flags: Flags, call_stack: MaterializedGraph) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate
        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)

        Returns
        -------
        pd.DataFrame
        """
        tflags = super().backward(flags, call_stack)
        out = {}
        for field in tflags.columns:
            flag_history = flags.history[field]
            flag_pos = flag_history.idxmax()
            flag_funcs = self._getFieldFunctions(field, call_stack)
            # NOTE:
            # we prepend empty strings to handle default columns in `Flags`
            # and potentially given flags not generated during the saqc run,
            # represented by `call_stack`
            flag_funcs = (
                [""] * (len(flag_history.hist.columns) - len(flag_funcs))
            ) + flag_funcs
            var_flags = {
                self._COL_LABELS["flag"]: tflags[field],
                self._COL_LABELS["comment"]: flag_pos.apply(
                    lambda p: json.dumps({"test": flag_funcs[p]})
                ),
                self._COL_LABELS["cause"]: "",
            }
            out[field] = pd.DataFrame(var_flags)
        return pd.concat(out, axis="columns")


class PositionalTranslator(Translator):

    """
    Implements the translation from and to the flagging scheme implemented by CHS
    """

    _FORWARD: Dict[int, float] = {0: UNFLAGGED, 1: DOUBTFUL, 2: BAD}
    _BACKWARD: Dict[float, int] = {
        UNTOUCHED: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def forward(self, flags: pd.DataFrame) -> Flags:
        """
        Translate from 'extrnal flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """
        data = {}
        for field in flags.columns:
            # drop the first column (i.e. the '9')
            fflags = pd.DataFrame(
                flags[field].apply(tuple).tolist(), index=flags[field].index
            ).iloc[:, 1:]

            tflags = super().forward(fflags.astype(int)).toFrame()
            tflags.insert(
                loc=0, column=0, value=pd.Series(UNFLAGGED, index=fflags.index)
            )
            data[field] = tflags
        return Flags(data)

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
            tflags = (
                thist.astype(int).astype(str).apply(lambda x: x.sum(), axis="columns")
            )
            # NOTE: work around the default first column history columns (see GL#182)
            out[field] = "9" + tflags.str.slice(start=1)
        return pd.DataFrame(out)
