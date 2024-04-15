#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import json
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
import pytest

from saqc.constants import BAD, DOUBTFUL, FILTER_NONE, UNFLAGGED
from saqc.core import Flags, SaQC
from saqc.core.translation import DmpScheme, MappingScheme, PositionalScheme
from saqc.core.translation.floatscheme import AnnotatedFloatScheme
from tests.common import initData


def _genTranslators():
    for dtype in (str, float, int):
        flags = {
            dtype(-2): UNFLAGGED,
            dtype(-1): BAD,
            **{dtype(f * 10): float(f) for f in range(10)},
        }
        scheme = MappingScheme(flags, {v: k for k, v in flags.items()})
        yield flags, scheme


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
    for flags, scheme in _genTranslators():
        for k, expected in flags.items():
            got = scheme(k)
            assert expected == got or np.isnan([got, expected]).all()

        for k in ["bad", 3.14, max]:
            with pytest.raises(ValueError):
                scheme(k)


def test_backwardTranslation():
    field = "var1"
    for _, scheme in _genTranslators():
        keys = tuple(scheme._backward.keys())
        flags = _genFlags({field: np.array(keys)})
        translated = scheme.toExternal(flags)
        expected = set(scheme._backward.values())
        assert not (set(translated[field]) - expected)


def test_backwardTranslationFail():
    field = "var1"
    for _, scheme in _genTranslators():
        keys = tuple(scheme._backward.keys())
        # add an scheme invalid value to the flags
        flags = _genFlags({field: np.array(keys + (max(keys) + 1,))})
        with pytest.raises(ValueError):
            scheme.toExternal(flags)


def test_dmpTranslator():
    scheme = DmpScheme()
    # generate a bunch of dummy flags
    keys = np.array(tuple(scheme._backward.keys()) * 50)
    flags = _genFlags({"var1": keys, "var2": keys, "var3": keys})
    flags[:, "var1"] = BAD
    flags[:, "var1"] = DOUBTFUL
    flags[:, "var2"] = BAD

    history1 = flags.history["var1"]
    history1.meta[1].update({"func": "flagFoo", "kwargs": {"cause": "AUTOFLAGGED"}})
    history1.meta[2].update({"func": "flagBar", "kwargs": {"comment": "I did it"}})

    history2 = flags.history["var2"]
    history2.meta[-1].update(
        {"func": "flagFoo", "kwargs": {"cause": "BELOW_OR_ABOVE_MIN_MAX"}}
    )

    tflags = scheme.toExternal(flags)

    for df in tflags.values():
        assert set(df.columns) == {
            "quality_flag",
            "quality_comment",
            "quality_cause",
        }

    assert (tflags["var1"]["quality_flag"] == "DOUBTFUL").all(axis=None)
    assert (
        tflags["var1"]["quality_comment"]
        == '{"test": "flagBar", "comment": "I did it"}'
    ).all(axis=None)

    assert (tflags["var1"]["quality_cause"] == "OTHER").all(axis=None)

    assert (tflags["var2"]["quality_flag"] == "BAD").all(axis=None)
    assert (
        tflags["var2"]["quality_comment"] == '{"test": "flagFoo", "comment": ""}'
    ).all(axis=None)
    assert (tflags["var2"]["quality_cause"] == "BELOW_OR_ABOVE_MIN_MAX").all(axis=None)

    assert (
        tflags["var3"].loc[flags["var3"] == BAD, "quality_comment"]
        == '{"test": "unknown", "comment": ""}'
    ).all(axis=None)
    assert (tflags["var3"].loc[flags["var3"] == BAD, "quality_cause"] == "OTHER").all(
        axis=None
    )
    assert (tflags["var3"].loc[flags["var3"] == UNFLAGGED, "quality_cause"] == "").all(
        axis=None
    )


def test_positionalTranslator():
    scheme = PositionalScheme()
    flags = _genFlags({"var1": np.zeros(100), "var2": np.zeros(50)})
    flags[1::3, "var1"] = BAD
    flags[1::3, "var1"] = DOUBTFUL
    flags[2::3, "var1"] = BAD

    tflags = scheme.toExternal(flags)
    assert (tflags["var2"].replace(-9999, np.nan).dropna() == 90).all(axis=None)
    assert (tflags["var1"].iloc[1::3] == 90210).all(axis=None)
    assert (tflags["var1"].iloc[2::3] == 90002).all(axis=None)


def test_positionalTranslatorIntegration():
    data = initData(3)
    col: str = data.columns[0]

    scheme = PositionalScheme()
    saqc = SaQC(data=data, scheme=scheme)
    saqc = saqc.flagMissing(col).flagRange(col, min=3, max=10, flag=DOUBTFUL)
    flags = saqc.flags

    for field in flags.keys():
        assert flags[field].astype(str).str.match("^9[012]*$").all()

    round_trip = scheme.toExternal(scheme.toInternal(flags))

    assert (flags.columns == round_trip.columns).all()
    for col in flags.columns:
        assert (flags[col] == round_trip[col]).all()
        assert (flags[col].index == round_trip[col].index).all()


def test_dmpTranslatorIntegration():
    data = initData(1)
    col = data.columns[0]

    scheme = DmpScheme()
    saqc = SaQC(data=data, scheme=scheme)
    saqc = saqc.flagMissing(col).flagRange(col, min=3, max=10)
    flags = saqc.flags

    qflags = pd.DataFrame({k: v["quality_flag"] for k, v in flags.items()})
    qfunc = pd.DataFrame({k: v["quality_comment"] for k, v in flags.items()})
    qcause = pd.DataFrame({k: v["quality_cause"] for k, v in flags.items()})

    assert qflags.isin(scheme._forward.keys()).all(axis=None)
    assert (
        qfunc.map(lambda v: json.loads(v)["test"] if v else "")
        .isin({"", "flagMissing", "flagRange"})
        .all(axis=None)
    )
    assert (qcause[qflags[col] == "BAD"] == "OTHER").all(axis=None)

    round_trip = scheme.toExternal(scheme.toInternal(flags))

    assert pd.DataFrame({k: v["quality_flag"] for k, v in round_trip.items()}).equals(
        qflags
    )
    assert pd.DataFrame(
        {k: v["quality_comment"] for k, v in round_trip.items()}
    ).equals(qfunc)
    assert pd.DataFrame({k: v["quality_cause"] for k, v in round_trip.items()}).equals(
        qcause
    )


def test_dmpValidCombinations():
    data = initData(1)
    col = data.columns[0]

    scheme = DmpScheme()
    saqc = SaQC(data=data, scheme=scheme)

    with pytest.raises(ValueError):
        saqc.flagRange(col, min=3, max=10, cause="SOMETHING_STUPID").flags

    with pytest.raises(ValueError):
        saqc.flagRange(col, min=3, max=10, cause="").flags


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
        saqc = saqc.flagRange(field=col, min=5, max=6, dfilter=FILTER_NONE).flagRange(
            col, min=3, max=10, dfilter=FILTER_NONE
        )
        flags = saqc._flags
        out.append(saqc)
    return out


def test_translationPreservesFlags():
    saqc1, saqc2 = _buildupSaQCObjects()
    flags1 = saqc1._flags
    flags2 = saqc2._flags

    for k in flags2.columns:
        got = flags2.history[k].hist

        f1hist = flags1.history[k].hist
        expected = pd.concat([f1hist, f1hist], axis="columns")
        expected.columns = got.columns

        assert expected.equals(got)


def test_multicallsPreserveHistory():
    saqc1, saqc2 = _buildupSaQCObjects()
    flags1 = saqc1._flags
    flags2 = saqc2._flags

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

    scheme = PositionalScheme()
    flags1 = saqc1._flags
    flags2 = saqc2._flags
    tflags1 = scheme.toExternal(flags1).astype(str)
    tflags2 = scheme.toExternal(flags2).astype(str)

    for k in flags2.columns:
        expected = tflags1[k].str.slice(start=1) * 2
        got = tflags2[k].str.slice(start=1)
        assert expected.equals(got)


def test_annotatedFloatScheme():
    data = initData(1)
    col = data.columns[0]

    scheme = AnnotatedFloatScheme()
    saqc = SaQC(data=data, scheme=scheme)
    saqc = saqc.setFlags(col, data=data[col].index[::4], flag=DOUBTFUL).flagRange(
        col, min=3, max=10, flag=BAD
    )
    flags = saqc.flags

    assert flags[col]["flag"].isin({DOUBTFUL, BAD, UNFLAGGED}).all(axis=None)
    assert flags[col]["func"].isin({"", "setFlags", "flagRange"}).all(axis=None)

    round_trip = scheme.toExternal(scheme.toInternal(flags))
    assert tuple(round_trip.keys()) == tuple(flags.keys())
    for key in flags.keys():
        assert round_trip[key].equals(flags[key])
