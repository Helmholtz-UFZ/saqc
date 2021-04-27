#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Union, Sequence

import numpy as np
import pandas as pd

import pytest

from dios import DictOfSeries

from saqc.constants import UNFLAGGED, BAD, DOUBTFUL
from saqc.core import translator
from saqc.core.translator import PositionalTranslator, Translator, DmpTranslator
from saqc.core.flags import Flags


def _genTranslators():
    for dtype in (str, float, int):
        flags = {
            dtype(-2): UNFLAGGED,
            dtype(-1): BAD,
            **{dtype(f * 10): float(f) for f in range(10)},
        }
        translator = Translator(flags)
        yield flags, translator


def _genFlags(data: Dict[str, Union[Sequence, pd.Series]]) -> Flags:

    flags = DictOfSeries()
    for k, v in data.items():
        if not isinstance(v, pd.Series):
            v = pd.Series(
                v, index=pd.date_range("2012-01-01", freq="1D", periods=len(v))
            )
        flags[k] = v

    return Flags(flags)


def test_forwardTranslation():
    for flags, translator in _genTranslators():
        for k, expected in flags.items():
            got = translator.forward(k)
            assert expected == got or np.isnan([got, expected]).all()

        for k in ["bad", 3.14, max]:
            with pytest.raises(ValueError):
                translator.forward(k)


def test_backwardTranslation():
    field = "var1"
    for _, translator in _genTranslators():
        keys = tuple(translator._backward.keys())
        flags = _genFlags({field: np.array(keys)})
        translated = translator.backward(flags, None)
        expected = set(translator._backward.values())
        assert not (set(translated[field]) - expected)


def test_backwardTranslationFail():
    field = "var1"
    for _, translator in _genTranslators():
        keys = tuple(translator._backward.keys())
        # add an scheme invalid value to the flags
        flags = _genFlags({field: np.array(keys + (max(keys) + 1,))})
        with pytest.raises(ValueError):
            translator.backward(flags, None)


@dataclass
class _Selector:
    field: str


@dataclass
class _Function:
    name: str


def test_dmpTranslator():
    translator = DmpTranslator()
    keys = np.array(tuple(translator._backward.keys()) * 50)
    flags = _genFlags({"var1": keys, "var2": keys})
    flags[:, "var1"] = BAD
    to_call = [
        (_Selector("var1"), None, _Function("flagFoo")),
        (_Selector("var1"), None, _Function("flagBar")),
        (_Selector("var2"), None, _Function("flagFoo")),
    ]
    tflags = translator.backward(flags, to_call)
    assert set(tflags.columns.get_level_values(1)) == {
        "quality_flag",
        "quality_comment",
        "quality_cause",
    }
    assert (tflags.loc[:, ("var1", "quality_comment")] == '{"test": "flagBar"}').all(
        axis=None
    )
    assert (tflags.loc[:, ("var2", "quality_comment")] == '{"test": "flagFoo"}').all(
        axis=None
    )


def test_positionalTranslator():
    translator = PositionalTranslator()
    flags = _genFlags({"var1": np.zeros(100), "var2": np.ones(50)})
    flags[1::3, "var1"] = BAD
    flags[1::3, "var1"] = DOUBTFUL
    flags[2::3, "var1"] = BAD

    tflags = translator.backward(flags, None)  # type: ignore
    assert (tflags["var2"].dropna() == "91").all(axis=None)
    assert (tflags["var1"].iloc[1::3] == "90210").all(axis=None)
    assert (tflags["var1"].iloc[2::3] == "90002").all(axis=None)


def test_positionalTranslatorIntegration():
    from tests.common import initData
    from saqc import SaQC

    data = initData(3)
    col = data.columns[0]

    saqc = SaQC(data=data, translator=PositionalTranslator())
    saqc = saqc.breaks.flagMissing(col, flag=2).outliers.flagRange(
        col, min=3, max=10, flag=2
    )
    data, flags = saqc.getResult()
