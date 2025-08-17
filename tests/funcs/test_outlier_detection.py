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
from saqc import BAD, UNFLAGGED, DictOfSeries, SaQC
from saqc.core.flags import initFlagsLike
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
        field, window="1h", method="modified", thresh=3.5, flag=BAD
    )
    flag_result = qc.flags[field]
    test_sum = (flag_result.iloc[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


def test_flagSpikesBasic(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagOffset(
        field, thresh=60, tolerance=10, window="20min", flag=BAD
    )
    flag_result = qc.flags[field]
    test_sum = (flag_result.iloc[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.slow
@pytest.mark.parametrize("dat", ["course_1", "course_2", "course_3", "course_4"])
def test_flagSpikesLimitRaise(dat, request):
    dat = request.getfixturevalue(dat)
    data, characteristics = dat()
    field, *_ = data.columns
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagRaise(
        field,
        thresh=2,
        freq="10min",
        raise_window="20min",
        flag=BAD,
    )
    assert np.all(qc.flags[field][characteristics["raise"]] > UNFLAGGED)
    assert not np.any(qc.flags[field][characteristics["return"]] > UNFLAGGED)
    assert not np.any(qc.flags[field][characteristics["drop"]] > UNFLAGGED)


# see test/functs/fixtures.py for the 'course_N'
def test_flagMVScores(course_3):
    def _check(fields, flags, characteristics):
        for field in fields:
            isflagged = flags[field] > UNFLAGGED
            assert isflagged[characteristics["raise"]].all()
            assert not isflagged[characteristics["return"]].any()
            assert not isflagged[characteristics["drop"]].any()

    data1, characteristics = course_3(
        periods=1000, initial_level=5, final_level=15, out_val=50
    )
    data2, characteristics = course_3(
        periods=1000, initial_level=20, final_level=1, out_val=30
    )
    fields = ["field1", "field2"]
    s1, s2 = data1["data"], data2["data"]
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = DictOfSeries(field1=s1, field2=s2)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags)
    qc = qc.processGeneric("field1", func=lambda x: np.log(x))
    qc = qc.processGeneric("field2", func=lambda x: np.log(x))
    qc = qc.assignKNNScore(
        ["field1", "field2"],
        target="kNNScores",
    )
    qc = qc.flagByStray("kNNScores", iter_start=0.95)
    qc = qc.transferFlags("kNNScores", target="field1")
    qc = qc.transferFlags("kNNScores", target="field2")

    _check(fields, qc.flags, characteristics)


def test_grubbs(course_3):
    data, char_dict = course_3(
        freq="10min",
        periods=45,
        initial_level=0,
        final_level=0,
        crowd_size=1,
        crowd_spacing=3,
        out_val=-10,
    )
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagUniLOF("data", density=0.4)
    assert np.all(qc.flags["data"][char_dict["drop"]] > UNFLAGGED)


@pytest.mark.parametrize(
    "parameters",
    [("standard", 1), ("modified", 1), ("modified", 3), ("standard", "3h")],
)
def test_flagCrossStatistics(parameters):
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
    qc = qc.flagZScore("data")

    assert (qc.flags.to_pandas().iloc[[5, 40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", min_residuals=10)

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
    qc = qc.flagZScore(["data", "data2"])
    assert (qc.flags.to_pandas().iloc[[5, 40, 80], 0] > 0).all()
    assert (qc.flags.to_pandas().iloc[[60], 1] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", min_residuals=10)

    assert (qc.flags.to_pandas()[["data", "data2"]] < 0).all().all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore(["data", "data2"], window="20D")

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()

    qc = saqc.SaQC(data)
    qc = qc.flagZScore("data", window=20)

    assert (qc.flags.to_pandas().iloc[[40, 80], 0] > 0).all()


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "cutoff", [{}, {"probability": 0.99}, {"thresh": "auto"}, {"thresh": 2}]
)
def test_flagUniLOF(spiky_data, n, p, cutoff):
    data = spiky_data[0]
    field, *_ = data.columns
    qc = SaQC(data).flagUniLOF(field, n=n, p=p, **cutoff)
    flag_result = qc.flags[field]
    test_sum = (flag_result.iloc[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


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
    test_sum = (flag_result.iloc[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


def test_flagOffsetRelative_basic_plateau():
    """Test basic plateau detection with bidirectional mode"""
    values = [100, 110, 100, 150, 150, 150, 109, 100, 50, 40, 105]
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="12h",
        thresh_relative=(0.2, -0.2),
    )
    actual_flags = qc.flags.to_pandas()

    # Expected: 150s plateau flagged (indices 3,4,5) and 50,40 dip flagged (indices 8,9)
    expected_flags = pd.DataFrame(
        {
            field: [
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
                BAD,
                BAD,
                BAD,
                UNFLAGGED,
                UNFLAGGED,
                BAD,
                BAD,
                UNFLAGGED,
            ]
        },
        index=idx,
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_single_spike():
    """Test single spike detection"""
    values = [100, 100, 130, 100, 100]  # Single upward spike
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="6h",
        thresh_relative=0.2,  # 20% threshold
    )
    actual_flags = qc.flags.to_pandas()

    # Expected: Single spike at index 2 flagged (130 is 30% above 100)
    expected_flags = pd.DataFrame(
        {field: [UNFLAGGED, UNFLAGGED, BAD, UNFLAGGED, UNFLAGGED]}, index=idx
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_no_return():
    """Test that spikes without returns are not flagged"""
    values = [100, 100, 150, 160, 170, 180]  # No return to baseline
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(field="sensor", window="6h", thresh_relative=0.2)
    actual_flags = qc.flags.to_pandas()

    # Expected: No flags since values don't return to baseline
    expected_flags = pd.DataFrame({field: [UNFLAGGED] * 6}, index=idx)

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_downward_only():
    """Test downward spike detection only"""
    values = [100, 100, 150, 100, 60, 100, 100]  # Upward and downward spikes
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="6h",
        thresh_relative=-0.3,  # Only 30% downward spikes,
    )
    actual_flags = qc.flags.to_pandas()

    # Expected: Only the 60 value flagged (40% drop), not the 150 (upward)
    expected_flags = pd.DataFrame(
        {
            field: [
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
                BAD,
                UNFLAGGED,
                UNFLAGGED,
            ]
        },
        index=idx,
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_threshold_boundary():
    """Test values right at threshold boundary"""
    values = [100, 121, 100, 120, 100]  # 21% vs 20% increases
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(field="sensor", window="6h", thresh_relative=0.2)
    actual_flags = qc.flags.to_pandas()

    print(actual_flags)

    expected_flags = pd.DataFrame(
        {field: [UNFLAGGED, BAD, UNFLAGGED, UNFLAGGED, UNFLAGGED]}, index=idx
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_window_timeout():
    """Test that spikes outside window are not flagged"""
    values = [100, 150, 150, 150, 150, 150, 150, 100]
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="3h",  # Short window, plateau is 6h long
        thresh_relative=0.2,
    )
    actual_flags = qc.flags.to_pandas()

    # Expected: No flags because return is outside 3h window
    expected_flags = pd.DataFrame({field: [UNFLAGGED] * 8}, index=idx)

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_multiple_plateaus():
    """Test multiple separate plateaus"""
    values = [100, 150, 150, 100, 100, 60, 60, 100, 100]  # Two separate plateaus
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="6h",
        thresh_relative=(0.3, -0.3),
        bidirectional=True,
    )
    actual_flags = qc.flags.to_pandas()

    expected_flags = pd.DataFrame(
        {
            field: [
                UNFLAGGED,
                BAD,
                BAD,
                UNFLAGGED,
                UNFLAGGED,
                BAD,
                BAD,
                UNFLAGGED,
                UNFLAGGED,
            ]
        },
        index=idx,
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_small_values():
    """Test with small baseline values"""
    values = [1.0, 1.5, 1.5, 1.0, 0.5, 1.0]  # Small values, same relative changes
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(field="sensor", window="6h", thresh_relative=(0.4, -0.4))
    actual_flags = qc.flags.to_pandas()

    # Expected: 1.5s flagged (50% increase), 0.5 flagged (50% decrease)
    expected_flags = pd.DataFrame(
        {field: [UNFLAGGED, BAD, BAD, UNFLAGGED, BAD, UNFLAGGED]}, index=idx
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_zero_baseline():
    """Test edge case with zero baseline"""
    values = [0, 5, 5, 0, 0]  # Spike from zero baseline
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(
        field="sensor",
        window="6h",
        thresh_relative=0.2,
    )
    actual_flags = qc.flags.to_pandas()
    expected_flags = pd.DataFrame(
        {
            field: [
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
                UNFLAGGED,
            ]
        },
        index=idx,
    )

    assert actual_flags.equals(expected_flags)


def test_flagOffsetRelative_negative_values():
    """Test with negative baseline values"""
    values = [-100, -150, -150, -100, -50, -100]  # Negative values
    idx = pd.date_range(start="2025-06-25", periods=len(values), freq="h")
    field = "sensor"
    data = pd.DataFrame({field: values}, index=idx)
    qc = SaQC(data).flagOffset(field="sensor", window="6h", thresh_relative=(0.3, -0.3))
    actual_flags = qc.flags.to_pandas()
    expected_flags = pd.DataFrame(
        {
            field: [
                UNFLAGGED,
                BAD,
                BAD,
                UNFLAGGED,
                BAD,
                UNFLAGGED,
            ]
        },
        index=idx,
    )
    assert actual_flags.equals(expected_flags)
