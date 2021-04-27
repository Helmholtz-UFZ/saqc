#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import namedtuple
from typing import Dict, Union, Sequence

import numpy as np
import pandas as pd

import pytest

from dios import DictOfSeries

from saqc.constants import UNFLAGGED, BAD, DOUBTFUL
from saqc.core.translator import PositionalTranslator, Translator, DmpTranslator
from saqc.core.flags import Flags
from saqc.core.core import SaQC

from tests.common import initData


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
            got = translator(k)
            assert expected == got or np.isnan([got, expected]).all()

        for k in ["bad", 3.14, max]:
            with pytest.raises(ValueError):
                translator(k)


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


def test_dmpTranslator():

    Selector = namedtuple("Selector", ["field"])
    Function = namedtuple("Function", ["name"])

    translator = DmpTranslator()
    keys = np.array(tuple(translator._backward.keys()) * 50)
    flags = _genFlags({"var1": keys, "var2": keys, "var3": keys})
    flags[:, "var1"] = BAD
    flags[:, "var1"] = DOUBTFUL
    flags[:, "var2"] = BAD
    to_call = [
        (Selector("var1"), Function("flagFoo")),
        (Selector("var1"), Function("flagBar")),
        (Selector("var2"), Function("flagFoo")),
    ]
    tflags = translator.backward(flags, to_call)
    assert set(tflags.columns.get_level_values(1)) == {
        "quality_flag",
        "quality_comment",
        "quality_cause",
    }

    assert (tflags.loc[:, ("var1", "quality_flag")] == "DOUBTFUL").all(axis=None)
    assert (tflags.loc[:, ("var1", "quality_comment")] == '{"test": "flagBar"}').all(
        axis=None
    )

    assert (tflags.loc[:, ("var2", "quality_flag")] == "BAD").all(axis=None)
    assert (tflags.loc[:, ("var2", "quality_comment")] == '{"test": "flagFoo"}').all(
        axis=None
    )

    assert (tflags.loc[:, ("var3", "quality_comment")] == '{"test": ""}').all(axis=None)


def test_positionalTranslator():
    translator = PositionalTranslator()
    flags = _genFlags({"var1": np.zeros(100), "var2": np.zeros(50)})
    flags[1::3, "var1"] = BAD
    flags[1::3, "var1"] = DOUBTFUL
    flags[2::3, "var1"] = BAD

    tflags = translator.backward(flags, None)  # type: ignore
    assert (tflags["var2"].dropna() == "9").all(axis=None)
    assert (tflags["var1"].iloc[1::3] == "9210").all(axis=None)
    assert (tflags["var1"].iloc[2::3] == "9002").all(axis=None)


def test_positionalTranslatorIntegration():

    data = initData(3)
    col: str = data.columns[0]

    translator = PositionalTranslator()
    saqc = SaQC(data=data, translator=translator)
    saqc = saqc.breaks.flagMissing(col).outliers.flagRange(col, min=3, max=10)
    data, flags = saqc.getResult()

    for field in flags.columns:
        assert flags[field].str.match("^9[012]*$").all()
    round_trip = translator.backward(translator.forward(flags), saqc._computed)

    assert (flags.values == round_trip.values).all()
    assert (flags.index == round_trip.index).all()
    assert (flags.columns == round_trip.columns).all()


def test_dmpTranslatorIntegration():

    data = initData(3)
    col = data.columns[0]

    translator = DmpTranslator()
    saqc = SaQC(data=data, translator=translator)
    saqc = saqc.breaks.flagMissing(col).outliers.flagRange(col, min=3, max=10)
    data, flags = saqc.getResult()

    qflags = flags.xs("quality_flag", axis="columns", level=1)
    qfunc = flags.xs("quality_comment", axis="columns", level=1).applymap(
        lambda v: json.loads(v)["test"]
    )
    qcause = flags.xs("quality_cause", axis="columns", level=1)

    assert qflags.isin(translator._forward.keys()).all(axis=None)
    assert qfunc.isin({"", "breaks.flagMissing", "outliers.flagRange"}).all(axis=None)
    assert (qcause == "").all(axis=None)

    round_trip = translator.backward(translator.forward(flags), saqc._computed)
    assert round_trip.xs("quality_flag", axis="columns", level=1).equals(qflags)
    assert (
        round_trip.xs("quality_comment", axis="columns", level=1)
        .applymap(lambda v: json.loads(v)["test"] == "")
        .all(axis=None)
    )


def _buildupSaQCObjects():

    """
    return two saqc object, whereas the flags from the previous run
    are reused
    """
    data = initData(3)
    col = data.columns[0]
    flags = None

    out = []
    for _ in range(2):
        saqc = SaQC(data=data, flags=flags)
        saqc = saqc.breaks.flagMissing(col, to_mask=False).outliers.flagRange(
            col, min=3, max=10, to_mask=False
        )
        saqc = saqc.evaluate()
        flags = saqc._flags
        out.append(saqc)
    return out


def test_translationPreservesFlags():

    saqc1, saqc2 = _buildupSaQCObjects()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)

    for k in flags2.columns:
        got = flags2.history[k].hist.iloc[:, 1:]

        f1hist = flags1.history[k].hist.iloc[:, 1:]
        expected = pd.concat([f1hist, f1hist], axis="columns")
        expected.columns = got.columns

        assert expected.equals(got)


def test_positionalTranslationPreservesFlags():

    saqc1, saqc2 = _buildupSaQCObjects()
    translator = PositionalTranslator()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)
    tflags1 = translator.backward(flags1, saqc1._computed)
    tflags2 = translator.backward(flags2, saqc2._computed)

    for k in flags2.columns:
        expected = tflags1[k].str.slice(start=1) * 2
        got = tflags2[k].str.slice(start=1)
        assert expected.equals(got)


def test_dmpTranslationPreservesFlags():

    saqc1, saqc2 = _buildupSaQCObjects()

    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)

    translator = DmpTranslator()
    tflags1 = translator.backward(flags1, saqc1._computed)
    tflags2 = translator.backward(flags2, saqc2._computed)

    assert tflags1.equals(tflags2)
