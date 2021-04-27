#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.core.flags import Flags, UNTOUCHED, UNFLAGGED, GOOD, DOUBTFUL, BAD
from saqc.core.lib import APIController, ColumnSelector
from saqc.core.register import SaQCFunction
from saqc.lib.types import UserFlag


CallStack = List[Tuple[ColumnSelector, APIController, SaQCFunction]]

# we need: {-np.inf, BAD} as translations
# tanslation schemes müssen gegeben werden, default: IdentityTranslator
# to_mask as part of th translator


class Translator:
    def __init__(
        self,
        forward: Dict[UserFlag, float],
        backward: Optional[Dict[float, UserFlag]] = None,
    ):
        # NOTE: we also add the keys to also allow the usage of internal flags
        self._forward = forward
        if backward is None:
            backward = {v: k for k, v in forward.items()}
        if {UNFLAGGED, BAD} - set(backward.keys()):
            raise ValueError(
                f"need translations for the special flags `UNFLAGGED` ({UNFLAGGED}) and `BAD` ({BAD})"
            )
        self._backward = backward

    def forward(self, flag: UserFlag) -> float:
        if flag not in self._forward:
            raise ValueError(f"invalid flag: {flag}")
        return self._forward[flag]

    def backward(self, flags: Flags, call_stack: CallStack) -> pd.DataFrame:
        # NOTE:
        # - we expect an instance of SaQC as child classes might
        #   need to access SaQC._to_call, but maybe the latter is sufficient?
        # - in theory `flags` should only contain valid values,
        #   as they all went through `Translator.forward` in practice:
        #   who knows...
        out = DictOfSeries()
        expected = pd.Index(self._backward.values())
        for field in flags.columns:
            out[field] = flags[field].replace(self._backward)
            # NOTE: for large datasets (> 100_000 values),
            # dropping first is signifacantly faster
            diff = pd.Index(out[field]).difference(expected)
            if not diff.empty:
                raise ValueError(f"flags were not translated: {expected.to_list()}")
        return out.to_df()


class FloatTranslator(Translator):
    def __init__(self):
        super().__init__(
            {-np.inf: -np.inf, **{k: k for k in np.arange(0, 256, dtype=float)}}
        )


class DmpTranslator(Translator):

    _FORWARD: Dict[str, float] = {
        "NIL": UNFLAGGED,
        "OK": GOOD,
        "DOUBTFUL": DOUBTFUL,
        "BAD": BAD,
    }
    _BACKWARD: Dict[float, str] = {v: k for k, v in _FORWARD.items()}

    def __init__(self):
        super().__init__(forward=self._FORWARD, backward=self._BACKWARD)

    def _getFieldFunctions(self, field: str, call_stack: CallStack) -> List[str]:
        # NOTE: `SaQC._to_call` shoul probably by an own class prviding such accessors.
        out = []
        for l, _, f in call_stack:
            if l.field == field:
                out.append(f.name)
        return out

    def backward(self, flags: Flags, call_stack: CallStack) -> pd.DataFrame:
        tflags = super().backward(flags, call_stack)
        out = {}
        for field in tflags.columns:
            flag_pos = flags.history[field].idxmax()
            flag_funcs = self._getFieldFunctions(field, call_stack)
            var_flags = {
                "quality_flag": tflags[field],
                "quality_comment": flag_pos.apply(
                    lambda p: json.dumps({"test": flag_funcs[p]})
                ),
                "quality_cause": "",
            }
            out[field] = pd.DataFrame(var_flags)
        out = pd.concat(out, axis="columns")
        return out


class PositionalTranslator(Translator):

    _FORWARD: Dict[int, float] = {0: UNFLAGGED, 1: DOUBTFUL, 2: BAD}
    _BACKWARD: Dict[float, int] = {
        UNTOUCHED: 0,
        UNFLAGGED: 0,
        GOOD: 0,
        DOUBTFUL: 1,
        BAD: 2,
    }

    def __init__(self):
        super().__init__(self._FORWARD, self._BACKWARD)

    def backward(self, flags: Flags, call_stack: CallStack) -> pd.DataFrame:
        out = {}
        for field in flags.columns:
            thist = flags.history[field].hist.replace(self._BACKWARD)
            out[field] = (
                thist.astype(int)
                .astype(str)
                .apply(lambda x: "9" + x.sum(), axis="columns")
            )
        return pd.DataFrame(out)
