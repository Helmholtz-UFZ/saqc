#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Dict, Union, Sequence

import numpy as np
import pandas as pd

import pytest

from dios import DictOfSeries

from saqc.constants import UNFLAGGED, BAD, DOUBTFUL
from saqc.core.translator import (
    PositionalTranslator,
    Translator,
    DmpTranslator,
)
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
        translator = Translator(flags, {v: k for k, v in flags.items()})
        yield flags, translator


def _genFlags(data: Dict[str, Union[Sequence, pd.Series]]) -> Flags:

    flags = Flags()
    for k, v in data.items():
        if not isinstance(v, pd.Series):
            v = pd.Series(
                v, index=pd.date_range("2012-01-01", freq="1D", periods=len(v))
            )
        flags[k] = v

    return flags


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
        translated = translator.backward(flags)
        expected = set(translator._backward.values())
        assert not (set(translated[field]) - expected)


def test_backwardTranslationFail():
    field = "var1"
    for _, translator in _genTranslators():
        keys = tuple(translator._backward.keys())
        # add an scheme invalid value to the flags
        flags = _genFlags({field: np.array(keys + (max(keys) + 1,))})
        with pytest.raises(ValueError):
            translator.backward(flags)


def test_dmpTranslator():

    translator = DmpTranslator()
    # generate a bunch of dummy flags
    keys = np.array(tuple(translator._backward.keys()) * 50)
    flags = _genFlags({"var1": keys, "var2": keys, "var3": keys})
    flags[:, "var1"] = BAD
    flags[:, "var1"] = DOUBTFUL
    flags[:, "var2"] = BAD

    history1 = flags.history["var1"]
    history1.meta[1].update({"func": "flagFoo", "keywords": {"cause": "AUTOFLAGGED"}})
    history1.meta[2].update({"func": "flagBar", "keywords": {"comment": "I did it"}})
    flags.history["var1"] = history1

    history2 = flags.history["var2"]
    history2.meta[-1].update(
        {"func": "flagFoo", "keywords": {"cause": "BELOW_OR_ABOVE_MIN_MAX"}}
    )
    flags.history["var2"] = history2

    tflags = translator.backward(flags)

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
    assert (tflags.loc[:, ("var2", "quality_cause")] == "BELOW_OR_ABOVE_MIN_MAX").all(
        axis=None
    )

    assert (
        tflags.loc[flags["var3"] == BAD, ("var3", "quality_comment")]
        == '{"test": "unknown", "comment": ""}'
    ).all(axis=None)
    assert (tflags.loc[flags["var3"] == BAD, ("var3", "quality_cause")] == "OTHER").all(
        axis=None
    )
    mask = flags["var3"] == UNFLAGGED
    assert (tflags.loc[mask, ("var3", "quality_cause")] == "").all(axis=None)


def test_positionalTranslator():
    translator = PositionalTranslator()
    flags = _genFlags({"var1": np.zeros(100), "var2": np.zeros(50)})
    flags[1::3, "var1"] = BAD
    flags[1::3, "var1"] = DOUBTFUL
    flags[2::3, "var1"] = BAD

    tflags = translator.backward(flags)
    assert (tflags["var2"].replace(-9999, np.nan).dropna() == 90).all(axis=None)
    assert (tflags["var1"].iloc[1::3] == 90210).all(axis=None)
    assert (tflags["var1"].iloc[2::3] == 90002).all(axis=None)


def test_positionalTranslatorIntegration():

    data = initData(3)
    col: str = data.columns[0]

    translator = PositionalTranslator()
    saqc = SaQC(data=data, scheme=translator)
    saqc = saqc.flagMissing(col).flagRange(col, min=3, max=10, flag=DOUBTFUL)
    data, flags = saqc.getResult()

    for field in flags.columns:
        assert flags[field].astype(str).str.match("^9[012]*$").all()

    round_trip = translator.backward(translator.forward(flags))

    assert (flags.values == round_trip.values).all()
    assert (flags.index == round_trip.index).all()
    assert (flags.columns == round_trip.columns).all()


def test_dmpTranslatorIntegration():

    data = initData(1)
    col = data.columns[0]

    translator = DmpTranslator()
    saqc = SaQC(data=data, scheme=translator)
    saqc = saqc.flagMissing(col).flagRange(col, min=3, max=10)
    data, flags = saqc.getResult()

    qflags = flags.xs("quality_flag", axis="columns", level=1)
    qfunc = flags.xs("quality_comment", axis="columns", level=1).applymap(
        lambda v: json.loads(v)["test"] if v else ""
    )
    qcause = flags.xs("quality_cause", axis="columns", level=1)

    assert qflags.isin(translator._forward.keys()).all(axis=None)
    assert qfunc.isin({"", "flagMissing", "flagRange"}).all(axis=None)
    assert (qcause[qflags[col] == "BAD"] == "OTHER").all(axis=None)

    round_trip = translator.backward(translator.forward(flags))

    assert round_trip.xs("quality_flag", axis="columns", level=1).equals(qflags)

    assert round_trip.xs("quality_comment", axis="columns", level=1).equals(
        flags.xs("quality_comment", axis="columns", level=1)
    )

    assert round_trip.xs("quality_cause", axis="columns", level=1).equals(
        flags.xs("quality_cause", axis="columns", level=1)
    )


def test_dmpValidCombinations():
    data = initData(1)
    col = data.columns[0]

    translator = DmpTranslator()
    saqc = SaQC(data=data, scheme=translator)

    with pytest.raises(ValueError):
        saqc.flagRange(col, min=3, max=10, cause="SOMETHING_STUPID").getResult()

    with pytest.raises(ValueError):
        saqc.flagRange(col, min=3, max=10, cause="").getResult()


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
        saqc = saqc.flagRange(field=col, min=5, max=6, to_mask=False).flagRange(
            col, min=3, max=10, to_mask=False
        )
        flags = saqc._flags
        out.append(saqc)
    return out


def test_translationPreservesFlags():

    saqc1, saqc2 = _buildupSaQCObjects()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)

    for k in flags2.columns:
        got = flags2.history[k].hist

        f1hist = flags1.history[k].hist
        expected = pd.concat([f1hist, f1hist], axis="columns")
        expected.columns = got.columns

        assert expected.equals(got)


def test_multicallsPreserveHistory():
    saqc1, saqc2 = _buildupSaQCObjects()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)

    # check, that the `History` is duplicated
    for col in flags2.columns:
        hist1 = flags1.history[col].hist
        hist2 = flags2.history[col].hist

        hist21 = hist2.iloc[:, : len(hist1.columns)]
        hist22 = hist2.iloc[:, len(hist1.columns) :]

        hist21.columns = hist1.columns
        hist22.columns = hist1.columns

        assert hist1.equals(hist21)
        assert hist1.equals(hist22)
        assert hist21.equals(hist22)


def test_positionalMulitcallsPreserveState():

    saqc1, saqc2 = _buildupSaQCObjects()

    translator = PositionalTranslator()
    _, flags1 = saqc1.getResult(raw=True)
    _, flags2 = saqc2.getResult(raw=True)
    tflags1 = translator.backward(flags1).astype(str)
    tflags2 = translator.backward(flags2).astype(str)

    for k in flags2.columns:
        expected = tflags1[k].str.slice(start=1) * 2
        got = tflags2[k].str.slice(start=1)
        assert expected.equals(got)
