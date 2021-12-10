#! /usr/bin/env python
# -*- coding: utf-8 -*-

import dios
import pandas as pd
import numpy as np
import saqc

from saqc.funcs.noise import flagByStatLowPass
from saqc.constants import *
from saqc.core import initFlagsLike
from saqc.funcs.drift import (
    flagDriftFromNorm,
    flagDriftFromReference,
)
from saqc.funcs.outliers import flagRange
from saqc.funcs.flagtools import flagManual, forceFlags, clearFlags
from saqc.funcs.tools import dropField, copyField, maskTime
from saqc.funcs.resampling import concatFlags
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
        data, flags = concatFlags(
            data, newfield, flags, method="match", target=field, flag=BAD
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

    _, flags = forceFlags(data, field, flags, flag=DOUBTFUL)
    assert all(flags[field] == DOUBTFUL)


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


def test_flagManual(data, field):
    flags = initFlagsLike(data)
    dat = data[field]

    mdata = pd.Series("lala", index=dat.index)
    index_exp = mdata.iloc[[10, 33, 200, 500]].index
    mdata.iloc[[101, 133, 220, 506]] = "b"
    mdata.loc[index_exp] = "a"
    shrinked = mdata.loc[index_exp.union(mdata.iloc[[1, 2, 3, 4, 600, 601]].index)]

    kwargs_list = [
        dict(mdata=mdata, mflag="a", method="plain", mformat="mflag", flag=BAD),
        dict(mdata=mdata, mflag="a", method="ontime", mformat="mflag", flag=BAD),
        dict(mdata=shrinked, mflag="a", method="ontime", mformat="mflag", flag=BAD),
    ]

    for kw in kwargs_list:
        _, fl = flagManual(data.copy(), field, flags.copy(), **kw)
        isflagged = fl[field] > UNFLAGGED
        assert isflagged[isflagged].index.equals(index_exp)

    # flag not exist in mdata
    _, fl = flagManual(
        data.copy(),
        field,
        flags.copy(),
        mdata=mdata,
        mflag="i do not exist",
        method="ontime",
        mformat="mflag",
        flag=BAD,
    )
    isflagged = fl[field] > UNFLAGGED
    assert isflagged[isflagged].index.equals(pd.DatetimeIndex([]))

    # check closure methods
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
    m_index = mdata.index
    flag_intervals = [
        (m_index[1], m_index[2]),
        (m_index[5], m_index[8]),
        (m_index[9], dat.index.shift(freq="1h")[-1]),
    ]
    bound_drops = {"right-open": [1], "left-open": [0], "closed": []}
    for method in ["right-open", "left-open", "closed"]:
        _, fl = flagManual(
            data.copy(),
            field,
            flags.copy(),
            mdata=mdata,
            mflag=1,
            method=method,
            mformat="mflag",
            flag=BAD,
        )
        isflagged = fl[field] > UNFLAGGED
        for flag_i in flag_intervals:
            f_i = isflagged[slice(flag_i[0], flag_i[-1])].index
            check_i = f_i.drop(
                [flag_i[k] for k in bound_drops[method]], errors="ignore"
            )
            assert isflagged[check_i].all()
            unflagged = isflagged[f_i.difference(check_i)]
            if not unflagged.empty:
                assert ~unflagged.all()


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_flagDriftFromNorm(dat):
    data = dat(periods=200, peak_level=5, name="field1")[0]
    data["field2"] = dat(periods=200, peak_level=10, name="field2")[0]["field2"]
    data["field3"] = dat(periods=200, peak_level=100, name="field3")[0]["field3"]

    fields = ["field1", "field2", "field3"]

    flags = initFlagsLike(data)
    _, flags_norm = flagDriftFromNorm(
        data=data.copy(),
        field=fields,
        flags=flags.copy(),
        freq="200min",
        spread=5,
        flag=BAD,
    )
    assert all(flags_norm["field3"] > UNFLAGGED)


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_flagDriftFromReference(dat):
    data = dat(periods=200, peak_level=5, name="field1")[0]
    data["field2"] = dat(periods=200, peak_level=10, name="field2")[0]["field2"]
    data["field3"] = dat(periods=200, peak_level=100, name="field3")[0]["field3"]

    fields = ["field1", "field2", "field3"]

    flags = initFlagsLike(data)

    _, flags_ref = flagDriftFromReference(
        data=data.copy(),
        field=fields,
        flags=flags.copy(),
        reference="field1",
        freq="3D",
        thresh=20,
        flag=BAD,
    )
    assert all(flags_ref["field3"] > UNFLAGGED)


def test_transferFlags():
    data = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    qc = saqc.SaQC(data)
    qc = qc.flagRange("a", max=1.5)
    qc = qc.transferFlags(["a", "a"], ["b", "c"])
    assert np.all(
        qc.flags["b"].values == np.array([saqc.constants.UNFLAGGED, saqc.constants.BAD])
    )
    assert np.all(
        qc.flags["c"].values == np.array([saqc.constants.UNFLAGGED, saqc.constants.BAD])
    )
