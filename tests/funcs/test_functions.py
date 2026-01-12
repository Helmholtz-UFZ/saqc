#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from saqc import BAD, DOUBTFUL, UNFLAGGED, DictOfSeries, SaQC
from saqc.core.flags import initFlagsLike
from tests.common import initData
from tests.fixtures import char_dict, course_1  # noqa, todo: fix fixtures


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


def test_statPass():
    data = pd.Series(0, index=pd.date_range("2000", "2001", freq="1D"), name="data")
    noise = [-1, 1] * 10
    data[100:120] = noise
    data[200:210] = noise[:10]
    data = DictOfSeries(data=data)
    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagByScatterLowpass(
        "data", "20D", 0.999, np.std, "5D", 0.999, 0, flag=BAD
    )
    assert (qc.flags["data"].iloc[:100] == UNFLAGGED).all()
    assert (qc.flags["data"].iloc[100:120] == BAD).all()
    assert (qc.flags["data"].iloc[121:] == UNFLAGGED).all()


def test_flagRange(data, field):
    min, max = 10, 90
    flags = initFlagsLike(data)
    qc = SaQC(data, flags)
    qc = qc.flagRange(field, min=min, max=max, flag=BAD)
    flagged = qc.flags[field] > UNFLAGGED
    expected = (data[field] < min) | (data[field] > max)
    assert all(flagged == expected)


def test_selectTime(data, field):
    data[field].iloc[::2] = 0
    data[field].iloc[1::2] = 50
    nyears = len(data[field].index.year.unique())

    tests = [
        (
            {
                "min": 1,
                "max": 100,
                "startmonth": 7,
                "startday": 1,
                "endmonth": 8,
                "endday": 31,
            },
            31 * 2 * nyears // 2,
        ),
        (
            {
                "min": 1,
                "max": 100,
                "startmonth": 12,
                "startday": 16,
                "endmonth": 1,
                "endday": 15,
            },
            31 * nyears // 2 + 1,
        ),
    ]

    flags = initFlagsLike(data)
    qc = SaQC(data, flags)
    for test, expected in tests:
        newfield = f"{field}_masked"
        start = f"{test['startmonth']:02}-{test['startday']:02}T00:00:00"
        end = f"{test['endmonth']:02}-{test['endday']:02}T00:00:00"

        qc = qc.copyField(field, newfield)
        qc = qc.selectTime(
            newfield,
            mode="periodic",
            start=start,
            end=end,
            closed=True,
            flag=BAD,
        )
        qc = qc.flagRange(newfield, min=test["min"], max=test["max"], flag=BAD)
        qc = qc.transferFlags(newfield, target=field, flag=BAD, overwrite=True)
        qc = qc.dropField(newfield)
        flagged = qc._flags[field] > UNFLAGGED
        assert flagged.sum() == expected


def test_clearFlags(data, field):
    flags = initFlagsLike(data)
    flags[:, field] = BAD
    assert all(flags[field] == BAD)

    qc = SaQC(data, flags)
    qc = qc.clearFlags(field)
    assert all(qc._flags[field] == UNFLAGGED)


def test_forceFlags(data, field):
    flags = initFlagsLike(data)
    flags[:, field] = BAD
    assert all(flags[field] == BAD)

    qc = SaQC(data, flags).forceFlags(field, flag=DOUBTFUL)
    assert all(qc._flags[field] == DOUBTFUL)


def test_flagIsolated(data, field):
    flags = initFlagsLike(data)
    d_len = len(data[field].index)
    data[field].iloc[1:3] = np.nan
    data[field].iloc[4:5] = np.nan
    flags[data[field].index[5:6], field] = BAD
    data[field].iloc[11:13] = np.nan
    data[field].iloc[15:17] = np.nan

    #              data  flags
    # 2016-01-01   0.0   -inf
    # 2016-01-02   NaN   -inf
    # 2016-01-03   NaN   -inf
    # 2016-01-04   3.0   -inf
    # 2016-01-05   NaN   -inf
    # 2016-01-06   5.0  255.0
    # 2016-01-07   6.0   -inf
    # 2016-01-08   7.0   -inf
    #         ..    ..     ..

    qc = SaQC(data, flags).flagIsolated(
        field, group_window="1D", gap_window="2.1D", flag=BAD
    )
    assert (qc._flags[field].iloc[[3, 5]] == BAD).all()
    neg_list = [k for k in range(d_len) if k not in [3, 5]]
    assert (qc._flags[field].iloc[neg_list] == UNFLAGGED).all()

    qc = qc.flagIsolated(
        field,
        group_window="2D",
        gap_window="2.1D",
        flag=BAD,
    )
    assert (qc._flags[field].iloc[[3, 5, 13, 14]] == BAD).all()
    neg_list = [k for k in range(d_len) if k not in [3, 5, 13, 14]]
    assert (qc._flags[field].iloc[neg_list] == UNFLAGGED).all()


def test_flagDriftFromNorm(course_1):
    data = course_1(periods=200, peak_level=5, name="field1")[0]
    data["field2"] = course_1(periods=200, peak_level=10, name="field2")[0]["field2"]
    data["field3"] = course_1(periods=200, peak_level=100, name="field3")[0]["field3"]

    fields = ["field1", "field2", "field3"]

    flags = initFlagsLike(data)
    qc = SaQC(data, flags).flagDriftFromNorm(
        field=fields,
        window="200min",
        spread=5,
        flag=BAD,
    )
    assert all(qc._flags["field3"] > UNFLAGGED)


def test_flagDriftFromReference(course_1):
    data = course_1(periods=200, peak_level=5, name="field1")[0]
    data["field2"] = course_1(periods=200, peak_level=10, name="field2")[0]["field2"]
    data["field3"] = course_1(periods=200, peak_level=100, name="field3")[0]["field3"]

    fields = ["field1", "field2", "field3"]

    flags = initFlagsLike(data)

    qc = SaQC(data, flags).flagDriftFromReference(
        field=fields,
        reference="field1",
        freq="3D",
        thresh=20,
        flag=BAD,
    )
    assert all(qc._flags["field3"] > UNFLAGGED)


def test_flagJumps():
    data = pd.DataFrame(
        {"a": [1, 1, 1, 1, 1, 6, 6, 6, 6, 6]},
        index=pd.date_range(start="2020-01-01", periods=10, freq="D"),
    )
    qc = SaQC(data=data)
    qc = qc.flagJumps(field="a", thresh=1, window="2D")
    assert qc.flags["a"].iloc[5] == BAD
    assert np.all(qc.flags["a"].values[:5] == UNFLAGGED) & np.all(
        qc.flags["a"].values[6:] == UNFLAGGED
    )
