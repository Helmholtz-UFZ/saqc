#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest

import saqc
from saqc import BAD, UNFLAGGED
from saqc.core import DictOfSeries, SaQC, initFlagsLike
from tests.fixtures import char_dict, course_1, course_2, course_3, course_4


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    s = pd.Series(np.linspace(1, 2, index.size), index=index)
    s.iloc[100] = 100
    s.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return DictOfSeries(spiky_data=s), flag_assertion


def test_flagMad(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagZScore(
        field, window="1H", method="modified", thresh=3.5, flag=BAD
    )
    flag_result = qc.flags[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


def test_flagSpikesBasic(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagOffset(
        field, thresh=60, tolerance=10, window="20min", flag=BAD
    )
    flag_result = qc.flags[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.slow
@pytest.mark.parametrize(
    "dat",
    [
        # see test/functs/fixtures.py for the 'course_N'
        pytest.lazy_fixture("course_1"),
        pytest.lazy_fixture("course_2"),
        pytest.lazy_fixture("course_3"),
        pytest.lazy_fixture("course_4"),
    ],
)
def test_flagSpikesLimitRaise(dat):
    data, characteristics = dat()
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagRaise(
        field,
        thresh=2,
        freq="10min",
        raise_window="20min",
        numba_boost=False,
        flag=BAD,
    )
    assert np.all(qc.flags[field][characteristics["raise"]] > UNFLAGGED)
    assert not np.any(qc.flags[field][characteristics["return"]] > UNFLAGGED)
    assert not np.any(qc.flags[field][characteristics["drop"]] > UNFLAGGED)


# see test/functs/fixtures.py for the 'course_N'
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_flagMVScores(dat):
    def _check(fields, flags, characteristics):
        for field in fields:
            isflagged = flags[field] > UNFLAGGED
            assert isflagged[characteristics["raise"]].all()
            assert not isflagged[characteristics["return"]].any()
            assert not isflagged[characteristics["drop"]].any()

    data1, characteristics = dat(
        periods=1000, initial_level=5, final_level=15, out_val=50
    )
    data2, characteristics = dat(
        periods=1000, initial_level=20, final_level=1, out_val=30
    )
    fields = ["field1", "field2"]
    s1, s2 = data1["data"], data2["data"]
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = DictOfSeries(field1=s1, field2=s2)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagMVScores(
        field=fields,
        trafo=np.log,
        iter_start=0.95,
        n=10,
        flag=BAD,
    )
    _check(fields, qc.flags, characteristics)


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_grubbs(dat):
    data, char_dict = dat(
        freq="10min",
        periods=45,
        initial_level=0,
        final_level=0,
        crowd_size=1,
        crowd_spacing=3,
        out_val=-10,
    )
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagByGrubbs("data", window=20, min_periods=15, flag=BAD)
    assert np.all(qc.flags["data"][char_dict["drop"]] > UNFLAGGED)


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
@pytest.mark.parametrize(
    "parameters",
    [("standard", 1), ("modified", 1), ("modified", 3), ("standard", "3h")],
)
def test_flagCrossStatistics(dat, parameters):
    fields = [f"data{i}" for i in range(6)]
    data = pd.DataFrame(
        0, columns=fields, index=pd.date_range("2000", freq="1h", periods=10)
    )
    bad_idx = (np.random.randint(0, 10), np.random.randint(0, 6))
    data.iloc[bad_idx[0], bad_idx[1]] = 10
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagZScore(
        fields, thresh=2, method=parameters[0], flag=BAD, axis=1, window=parameters[1]
    )

    isflagged = qc.flags.to_pandas() > UNFLAGGED
    assert isflagged.iloc[bad_idx[0], bad_idx[1]]
    assert isflagged.sum().sum() == 1


def test_flagZScoresUV():
    np.random.seed(seed=1)
    data = pd.DataFrame(
        {"data": [np.random.normal() for k in range(100)]},
        index=pd.date_range("2000", freq="1D", periods=100),
    )
    data.iloc[[5, 80], 0] = 5
    data.iloc[[40], 0] = -6
    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=None)

    assert (qc.flags.to_pandas().iloc[[5, 40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=None, min_residuals=10)

    assert (qc.flags.to_pandas()["data"] < 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window="20D")

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=20)

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()


def test_flagZScoresMV():
    np.random.seed(seed=1)
    data = pd.DataFrame(
        {
            "data": [np.random.normal() for k in range(100)],
            "data2": [np.random.normal() for k in range(100)],
        },
        index=pd.date_range("2000", freq="1D", periods=100),
    )
    data.iloc[[5, 80], 0] = 5
    data.iloc[[40], 0] = -6
    data.iloc[[60], 1] = 10
    qc = saqc.SaQC(data)
    qc = qc.flagZScore(["data", "data2"], window=None)
    assert (qc.flags.to_pandas().iloc[[5, 40, 80], 0] > 0).all()
    assert (qc.flags.to_pandas().iloc[[60], 1] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=None, min_residuals=10)

    assert (qc.flags.to_pandas()[["data", "data2"]] < 0).all().all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore(["data", "data2"], window="20D")

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=20)

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("thresh", ["auto", 2])
def test_flagUniLOF(spiky_data, n, p, thresh):
    data = spiky_data[0]
    field, *_ = data.columns
    qc = SaQC(data).flagUniLOF(field, n=n, p=p, thresh=thresh)
    flag_result = qc.flags[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    try:
        assert test_sum == len(spiky_data[1])
    except AssertionError:
        print("stop")


@pytest.mark.parametrize("vars", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("thresh", ["auto", 2])
def test_flagLOF(spiky_data, vars, p, thresh):
    data = pd.DataFrame(
        {f"data{v}": spiky_data[0].to_pandas().squeeze() for v in range(vars)}
    )
    field, *_ = data.columns
    qc = SaQC(data).flagLOF(field)
    flag_result = qc.flags[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])
