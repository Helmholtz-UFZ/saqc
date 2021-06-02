#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Dict, Union, Sequence

import numpy as np
import pandas as pd

import pytest

from dios import DictOfSeries

from saqc.constants import UNFLAGGED, BAD, DOUBTFUL
from saqc.core import translator
from saqc.core.translator import (
    FloatTranslator,
    PositionalTranslator,
    Translator,
    DmpTranslator,
)
from saqc.core.flags import Flags
from saqc.core.core import SaQC
from saqc.core.lib import APIController, SaQCFunction, ColumnSelector
from saqc.funcs.flagtools import flagDummy

from tests.common import initData


def _genTranslators():
    for dtype in (str, float, int):
        flags = {
            dtype(-2): UNFLAGGED,
            dtype(-1): BAD,
            **{dtype(f * 10): float(f) for f in range(10)},
        }
        translator = Translator(flags, {v: k for k, v in flags.items()})
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


@pytest.mark.skip(reason="dmp translator implementation is currently blocked")
def test_dmpTranslator():

    translator = DmpTranslator()
    keys = np.array(tuple(translator._backward.keys()) * 50)
    flags = _genFlags({"var1": keys, "var2": keys, "var3": keys})
    flags[:, "var1"] = BAD
    flags[:, "var1"] = DOUBTFUL
    flags[:, "var2"] = BAD

    to_call = [
        # the initial columns
        (ColumnSelector("var1"), SaQCFunction("flagInit", flagDummy)),
        (ColumnSelector("var2"), SaQCFunction("flagInit", flagDummy)),
        (
            ColumnSelector("var3"),
            SaQCFunction("flagInit", flagDummy, comment="initial flags"),
        ),
        (ColumnSelector("var1"), SaQCFunction("flagFoo", flagDummy)),
        (
            ColumnSelector("var1"),
            SaQCFunction("flagBar", flagDummy, comment="I did it"),
        ),
        (ColumnSelector("var2"), SaQCFunction("flagFoo", flagDummy)),
    ]
    tflags = translator.backward(flags, to_call)
    assert set(tflags.columns.get_level_values(1)) == {
        "quality_flag",
        "quality_comment",
        "quality_cause",
    }

    assert (tflags.loc[:, ("var1", "quality_flag")] == "DOUBTFUL").all(axis=None)
    assert (
        tflags.loc[:, ("var1", "quality_comment")]
        == '{"test": "flagBar", "comment": "I did it"}'
    ).all(axis=None)
    assert (tflags.loc[:, ("var1", "quality_cause")] == "OTHER").all(axis=None)

    assert (tflags.loc[:, ("var2", "quality_flag")] == "BAD").all(axis=None)
    assert (
        tflags.loc[:, ("var2", "quality_comment")]
        == '{"test": "flagFoo", "comment": ""}'
    ).all(axis=None)
    assert (tflags.loc[:, ("var2", "quality_cause")] == "OTHER").all(axis=None)

    assert (
        tflags.loc[flags["var3"] == BAD, ("var3", "quality_comment")]
        == '{"test": "flagInit", "comment": "initial flags"}'
    ).all(axis=None)
    assert (tflags.loc[flags["var3"] == BAD, ("var3", "quality_cause")] == "OTHER").all(
        axis=None
    )
    assert (tflags.loc[flags["var3"] < DOUBTFUL, ("var3", "quality_cause")] == "").all(
        axis=None
    )


def test_positionalTranslator():
    translator = PositionalTranslator()
    flags = _genFlags({"var1": np.zeros(100), "var2": np.zeros(50)})
    flags[1::3, "var1"] = BAD
    flags[1::3, "var1"] = DOUBTFUL
    flags[2::3, "var1"] = BAD

    tflags = translator.backward(flags, None)  # type: ignore
    assert (tflags["var2"].replace(-9999, np.nan).dropna() == 90).all(axis=None)
    assert (tflags["var1"].iloc[1::3] == 90210).all(axis=None)
    assert (tflags["var1"].iloc[2::3] == 90002).all(axis=None)


def test_positionalTranslatorIntegration():

    data = initData(3)
    col: str = data.columns[0]

    translator = PositionalTranslator()
    saqc = SaQC(data=data, scheme=translator)
    saqc = saqc.breaks.flagMissing(col).outliers.flagRange(
        col, min=3, max=10, flag=DOUBTFUL
    )
    data, flags = saqc.getResult()

    for field in flags.columns:
        assert flags[field].astype(str).str.match("^9[012]*$").all()

    round_trip = translator.backward(*translator.forward(flags))

    assert (flags.values == round_trip.values).all()
    assert (flags.index == round_trip.index).all()
    assert (flags.columns == round_trip.columns).all()


@pytest.mark.skip(reason="dmp translator implementation is currently blocked")
def test_dmpTranslatorIntegration():

    data = initData(1)
    col = data.columns[0]

    translator = DmpTranslator()
    saqc = SaQC(data=data, scheme=translator)
    saqc = saqc.breaks.flagMissing(col).outliers.flagRange(col, min=3, max=10)
    data, flags = saqc.getResult()

    qflags = flags.xs("quality_flag", axis="columns", level=1)
    qfunc = flags.xs("quality_comment", axis="columns", level=1).applymap(
        lambda v: json.loads(v)["test"]
    )
    qcause = flags.xs("quality_cause", axis="columns", level=1)

    assert qflags.isin(translator._forward.keys()).all(axis=None)
    assert qfunc.isin({"", "breaks.flagMissing", "outliers.flagRange"}).all(axis=None)
    assert (qcause[qflags[col] == "BAD"] == "OTHER").all(axis=None)

    round_trip = translator.backward(*translator.forward(flags))

    assert round_trip.xs("quality_flag", axis="columns", level=1).equals(qflags)

    assert round_trip.xs("quality_comment", axis="columns", level=1).equals(
        flags.xs("quality_comment", axis="columns", level=1)
    )

    assert round_trip.xs("quality_cause", axis="columns", level=1).equals(
        flags.xs("quality_cause", axis="columns", level=1)
    )


@pytest.mark.skip(reason="dmp translator implementation is currently blocked")
def test_dmpValidCause():
    data = initData(1)
    col = data.columns[0]

    translator = DmpTranslator()
    saqc = SaQC(data=data, scheme=translator)
    saqc = saqc.outliers.flagRange(col, min=3, max=10, cause="SOMETHING_STUPID")
    with pytest.raises(ValueError):
        data, flags = saqc.getResult()

    saqc = saqc.outliers.flagRange(col, min=3, max=10, cause="BELOW_OR_ABOVE_MIN_MAX")
    data, flags = saqc.getResult()
    qflags = flags.xs("quality_flag", axis="columns", level=1)
    qcause = flags.xs("quality_cause", axis="columns", level=1)
    assert (qcause[qflags[col] == "BAD"] == "BELOW_OR_ABOVE_MIN_MAX").all(axis=None)
    assert (qcause[qflags[col] != "BAD"] == "").all(axis=None)


def _buildupSaQCObjects():

    """
    return two evaluated saqc objects calling the same functions,
    whereas the flags from the evaluetion of the first objetc are
    used as input flags of the second
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


def test_callHistoryYieldsSameResults():

    # a simple SaQC run
    data = initData(3)
    col = data.columns[0]
    saqc1 = SaQC(data=data)
    saqc1 = saqc1.breaks.flagMissing(col, to_mask=False).outliers.flagRange(
        col, min=3, max=10, to_mask=False
    )
    _, flags1 = saqc1.getResult(raw=True)

    # generate a dummy call history from flags
    translator = FloatTranslator()
    graph = translator.buildGraph(flags1)
    saqc2 = SaQC(data=data)

    # convert the call history into an excution plan and inject into a blank SaQC object
    saqc2._planned = [(s, APIController(), f) for s, f in graph]
    # replay the functions
    _, flags2 = saqc2.getResult()

    assert flags2.equals(flags1.toFrame())


def test_multicallsPreserveHistory():
    saqc1, saqc2 = _buildupSaQCObjects()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)

    # check, that the `History` is duplicated
    for col in flags2.columns:
        hist1 = flags1.history[col].hist.loc[:, 1:]
        hist2 = flags2.history[col].hist.loc[:, 1:]

        hist21 = hist2.iloc[:, : len(hist1.columns)]
        hist22 = hist2.iloc[:, len(hist1.columns) :]

        hist21.columns = hist1.columns
        hist22.columns = hist1.columns

        assert hist1.equals(hist21)
        assert hist1.equals(hist22)
        assert hist21.equals(hist22)

    assert len(saqc2._computed) == len(saqc1._computed) * 2


def test_positionalMulitcallsPreserveState():

    saqc1, saqc2 = _buildupSaQCObjects()

    translator = PositionalTranslator()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)
    tflags1 = translator.backward(flags1, saqc1._computed).astype(str)
    tflags2 = translator.backward(flags2, saqc2._computed).astype(str)

    for k in flags2.columns:
        expected = tflags1[k].str.slice(start=1) * 2
        got = tflags2[k].str.slice(start=1)
        assert expected.equals(got)


@pytest.mark.skip(reason="dmp translator implementation is currently blocked")
def test_smpTranslatorHandlesRenames():

    data = initData(3)

    this: str = data.columns[0]
    other: str = this + "_new"

    saqc = (
        SaQC(data=data)
        .outliers.flagRange(this, min=1, max=10)
        .tools.rename(this, other)
        .breaks.flagMissing(other, min=4, max=6)
    )
    saqc = saqc.evaluate()

    this_funcs = DmpTranslator._getFieldFunctions(this, saqc._computed)
    other_funcs = DmpTranslator._getFieldFunctions(other, saqc._computed)

    assert [f.name for f in this_funcs] == [
        "",
        "outliers.flagRange",
        "tools.rename",
        "breaks.flagMissing",
    ]

    # we skip the first function in both lists, as they are dummy functions
    # inserted to allow a proper replay of all function calls
    assert this_funcs[1:] == other_funcs[1:]
