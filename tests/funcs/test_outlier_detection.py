#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest

import dios
import saqc
from saqc.constants import BAD, UNFLAGGED
from saqc.core import SaQC, initFlagsLike
from tests.fixtures import char_dict, course_1, course_2, course_3, course_4


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    s = pd.Series(np.linspace(1, 2, index.size), index=index, name="spiky_data")
    s.iloc[100] = 100
    s.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return dios.DictOfSeries(s), flag_assertion


def test_flagMad(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagMAD(field, "1H", flag=BAD)
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
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["field1", "field2"])
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
def test_flagCrossStatistics(dat):
    data1, characteristics = dat(initial_level=0, final_level=0, out_val=0)
    data2, characteristics = dat(initial_level=0, final_level=0, out_val=10)
    fields = ["field1", "field2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["field1", "field2"])
    flags = initFlagsLike(data)

    qc = SaQC(data, flags).flagCrossStatistics(
        fields, thresh=3, method=np.mean, flag=BAD
    )
    for field in fields:
        isflagged = qc.flags[field] > UNFLAGGED
        assert isflagged[characteristics["raise"]].all()


def test_flagZScores():
    np.random.seed(seed=1)
    data = pd.Series(
        [np.random.normal() for k in range(100)],
        index=pd.date_range("2000", freq="1D", periods=100),
        name="data",
    )
    data.iloc[[5, 80]] = 5
    data.iloc[[40]] = -6
    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=None)

    assert (qc.flags.to_df().iloc[[5, 40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=None, min_residuals=10)

    assert (qc.flags.to_df()["data"] < 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window="20D")

    assert (qc.flags.to_df().iloc[[40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=20)

    assert (qc.flags.to_df().iloc[[40, 80], 0] > 0).all()
