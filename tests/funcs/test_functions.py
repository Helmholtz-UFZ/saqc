#! /usr/bin/env python
# -*- coding: utf-8 -*-

import dios
import pandas as pd
import numpy as np

from saqc.funcs.noise import flagByStatLowPass
from saqc.constants import *
from saqc.core import initFlagsLike
from saqc.funcs.drift import (
    flagDriftFromNorm,
    flagDriftFromReference,
    flagDriftFromScaledNorm,
)
from saqc.funcs.outliers import flagCrossStatistic, flagRange
from saqc.funcs.flagtools import flagManual, forceFlags, clearFlags
from saqc.funcs.tools import dropField, copyField, maskTime
from saqc.funcs.resampling import reindexFlags
from saqc.funcs.breaks import flagIsolated

from tests.fixtures import *
from tests.common import initData


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
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    data, flags = flagByStatLowPass(
        data, "data", flags, np.std, "20D", 0.999, "5D", 0.999, 0, flag=BAD
    )
    assert (flags["data"].iloc[:100] == UNFLAGGED).all()
    assert (flags["data"].iloc[100:120] == BAD).all()
    assert (flags["data"].iloc[121:] == UNFLAGGED).all()


def test_flagRange(data, field):
    min, max = 10, 90
    flags = initFlagsLike(data)
    data, flags = flagRange(data, field, flags, min=min, max=max, flag=BAD)
    flagged = flags[field] > UNFLAGGED
    expected = (data[field] < min) | (data[field] > max)
    assert all(flagged == expected)


def test_flagSesonalRange(data, field):
    data.iloc[::2] = 0
    data.iloc[1::2] = 50
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

    for test, expected in tests:
        flags = initFlagsLike(data)
        newfield = f"{field}_masked"
        start = f"{test['startmonth']:02}-{test['startday']:02}T00:00:00"
        end = f"{test['endmonth']:02}-{test['endday']:02}T00:00:00"

        data, flags = copyField(data, field, flags, field + "_masked")
        data, flags = maskTime(
            data,
            newfield,
            flags,
            mode="periodic",
            start=start,
            end=end,
            closed=True,
            flag=BAD,
        )
        data, flags = flagRange(
            data, newfield, flags, min=test["min"], max=test["max"], flag=BAD
        )
        data, flags = reindexFlags(
            data, field, flags, method="match", source=newfield, flag=BAD
        )
        data, flags = dropField(data, newfield, flags)
        flagged = flags[field] > UNFLAGGED
        assert flagged.sum() == expected


def test_clearFlags(data, field):
    flags = initFlagsLike(data)
    flags[:, field] = BAD
    assert all(flags[field] == BAD)

    _, flags = clearFlags(data, field, flags)
    assert all(flags[field] == UNFLAGGED)


def test_forceFlags(data, field):
    flags = initFlagsLike(data)
    flags[:, field] = BAD
    assert all(flags[field] == BAD)

    _, flags = forceFlags(data, field, flags, flag=DOUBT)
    assert all(flags[field] == DOUBT)


def test_flagIsolated(data, field):
    flags = initFlagsLike(data)

    data.iloc[1:3, 0] = np.nan
    data.iloc[4:5, 0] = np.nan
    flags[data[field].index[5:6], field] = BAD
    data.iloc[11:13, 0] = np.nan
    data.iloc[15:17, 0] = np.nan

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

    _, flags_result = flagIsolated(
        data, field, flags, group_window="1D", gap_window="2.1D", flag=BAD
    )

    assert flags_result[field].iloc[[3, 5]].all()

    data, flags_result = flagIsolated(
        data,
        field,
        flags_result,
        group_window="2D",
        gap_window="2.1D",
        continuation_range="1.1D",
        flag=BAD,
    )
    assert flags_result[field].iloc[[3, 5, 13, 14]].all()


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_flagCrossScoring(dat):
    data1, characteristics = dat(initial_level=0, final_level=0, out_val=0)
    data2, characteristics = dat(initial_level=0, final_level=0, out_val=10)
    field = "dummy"
    fields = ["data1", "data2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["data1", "data2"])
    flags = initFlagsLike(data)
    _, flags_result = flagCrossStatistic(
        data, field, flags, fields=fields, thresh=3, method=np.mean, flag=BAD
    )
    for field in fields:
        isflagged = flags_result[field] > UNFLAGGED
        assert isflagged[characteristics["raise"]].all()


def test_flagManual(data, field):
    flags = initFlagsLike(data)
    args = data, field, flags
    dat = data[field]

    mdata = pd.Series("lala", index=dat.index)
    index_exp = mdata.iloc[[10, 33, 200, 500]].index
    mdata.iloc[[101, 133, 220, 506]] = "b"
    mdata.loc[index_exp] = "a"
    shrinked = mdata.loc[index_exp.union(mdata.iloc[[1, 2, 3, 4, 600, 601]].index)]

    kwargs_list = [
        dict(mdata=mdata, mflag="a", method="plain", flag=BAD),
        dict(mdata=mdata.to_list(), mflag="a", method="plain", flag=BAD),
        dict(mdata=mdata, mflag="a", method="ontime", flag=BAD),
        dict(mdata=shrinked, mflag="a", method="ontime", flag=BAD),
    ]

    for kw in kwargs_list:
        _, fl = flagManual(*args, **kw)
        isflagged = fl[field] > UNFLAGGED
        assert isflagged[isflagged].index.equals(index_exp)

    # flag not exist in mdata
    _, fl = flagManual(
        *args, mdata=mdata, mflag="i do not exist", method="ontime", flag=BAD
    )
    isflagged = fl[field] > UNFLAGGED
    assert isflagged[isflagged].index.equals(pd.DatetimeIndex([]))

    # check right-open / ffill
    index = pd.date_range(start="2016-01-01", end="2018-12-31", periods=11)
    mdata = pd.Series(0, index=index)
    mdata.loc[index[[1, 5, 6, 7, 9, 10]]] = 1
    # >>> mdata
    # 2016-01-01 00:00:00    0
    # 2016-04-19 12:00:00    1
    # 2016-08-07 00:00:00    0
    # 2016-11-24 12:00:00    0
    # 2017-03-14 00:00:00    0
    # 2017-07-01 12:00:00    1
    # 2017-10-19 00:00:00    1
    # 2018-02-05 12:00:00    1
    # 2018-05-26 00:00:00    0
    # 2018-09-12 12:00:00    1
    # 2018-12-31 00:00:00    1
    # dtype: int64

    # add first and last index from data
    expected = mdata.copy()
    expected.loc[dat.index[0]] = 0
    expected.loc[dat.index[-1]] = 1
    expected = expected.astype(bool)

    _, fl = flagManual(*args, mdata=mdata, mflag=1, method="right-open", flag=BAD)
    isflagged = fl[field] > UNFLAGGED
    last = expected.index[0]

    for curr in expected.index[1:]:
        expected_value = mdata[last]
        # datetime slicing is inclusive !
        i = isflagged[last:curr].index[:-1]
        chunk = isflagged.loc[i]
        assert (chunk == expected_value).all()
        last = curr
    # check last value
    assert isflagged[curr] == expected[curr]

    # check left-open / bfill
    expected.loc[dat.index[-1]] = 0  # this time the last is False
    _, fl = flagManual(*args, mdata=mdata, mflag=1, method="left-open", flag=BAD)
    isflagged = fl[field] > UNFLAGGED
    last = expected.index[0]
    assert isflagged[last] == expected[last]

    for curr in expected.index[1:]:
        expected_value = mdata[curr]
        # datetime slicing is inclusive !
        i = isflagged[last:curr].index[1:]
        chunk = isflagged.loc[i]
        assert (chunk == expected_value).all()
        last = curr


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_flagDriftFromNormal(dat):
    data = dat(periods=200, peak_level=5, name="d1")[0]
    data["d2"] = dat(periods=200, peak_level=10, name="d2")[0]["d2"]
    data["d3"] = dat(periods=200, peak_level=100, name="d3")[0]["d3"]
    data["d4"] = 3 + 4 * data["d1"]
    data["d5"] = 3 + 4 * data["d1"]

    flags = initFlagsLike(data)
    data_norm, flags_norm = flagDriftFromNorm(
        data,
        "dummy",
        flags,
        ["d1", "d2", "d3"],
        freq="200min",
        spread=5,
        flag=BAD,
    )

    data_ref, flags_ref = flagDriftFromReference(
        data,
        "d1",
        flags,
        ["d1", "d2", "d3"],
        freq="3D",
        thresh=20,
        flag=BAD,
    )

    data_scale, flags_scale = flagDriftFromScaledNorm(
        data,
        "dummy",
        flags,
        ["d1", "d3"],
        ["d4", "d5"],
        freq="3D",
        thresh=20,
        spread=5,
        flag=BAD,
    )
    assert all(flags_norm["d3"] > UNFLAGGED)
    assert all(flags_ref["d3"] > UNFLAGGED)
    assert all(flags_scale["d3"] > UNFLAGGED)
