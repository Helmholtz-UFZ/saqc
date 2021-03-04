#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
import dios

from saqc.common import *
from saqc.flagger import Flagger, initFlagsLike
from saqc.funcs.drift import flagDriftFromNorm, flagDriftFromReference, flagDriftFromScaledNorm
from saqc.funcs.outliers import flagCrossStatistic, flagRange
from saqc.funcs.flagtools import flagManual, forceFlags, clearFlags
from saqc.funcs.tools import drop, copy, mask
from saqc.funcs.resampling import reindexFlags
from saqc.funcs.breaks import flagIsolated
from test.common import initData, TESTFLAGGER


@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


def test_flagRange(data, field):
    min, max = 10, 90
    flagger = initFlagsLike(data)
    data, flagger = flagRange(data, field, flagger, min=min, max=max, flag=BAD)
    flagged = flagger[field] > UNFLAGGED
    expected = (data[field] < min) | (data[field] > max)
    assert all(flagged == expected)


def test_flagSesonalRange(data, field):
    # prepare
    data.iloc[::2] = 0
    data.iloc[1::2] = 50
    nyears = len(data[field].index.year.unique())

    tests = [
        ({"min": 1, "max": 100, "startmonth": 7, "startday": 1, "endmonth": 8, "endday": 31, }, 31 * 2 * nyears // 2,),
        (
        {"min": 1, "max": 100, "startmonth": 12, "startday": 16, "endmonth": 1, "endday": 15, }, 31 * nyears // 2 + 1,),
    ]

    for test, expected in tests:
        flagger = initFlagsLike(data)
        newfield = f"{field}_masked"
        start = f"{test['startmonth']:02}-{test['startday']:02}T00:00:00"
        end = f"{test['endmonth']:02}-{test['endday']:02}T00:00:00"

        data, flagger = copy(data, field, flagger, field + "_masked")
        data, flagger = mask(
            data, newfield, flagger,
            mode='periodic', period_start=start, period_end=end, include_bounds=True, flag=BAD
        )
        data, flagger = flagRange(data, newfield, flagger, min=test['min'], max=test['max'], flag=BAD)
        data, flagger = reindexFlags(data, field, flagger, method='match', source=newfield, flag=BAD)
        data, flagger = drop(data, newfield, flagger)
        flagged = flagger[field] > UNFLAGGED
        assert flagged.sum() == expected


def test_clearFlags(data, field):
    flagger = initFlagsLike(data)
    flagger[:, field] = BAD
    assert all(flagger[field] == BAD)

    _, flagger = clearFlags(data, field, flagger)
    assert all(flagger[field] == UNFLAGGED)


def test_forceFlags(data, field):
    flagger = initFlagsLike(data)
    flagger[:, field] = BAD
    assert all(flagger[field] == BAD)

    _, flagger = forceFlags(data, field, flagger, flag=DOUBT)
    assert all(flagger[field] == DOUBT)


# todo: @luenensc: i dont get the test -- palmb
def test_flagIsolated(data, field):
    flagger = initFlagsLike(data)

    data.iloc[1:3, 0] = np.nan
    data.iloc[4:5, 0] = np.nan
    data.iloc[11:13, 0] = np.nan
    data.iloc[15:17, 0] = np.nan

    s = data[field].iloc[5:6]
    flagger[s.index, field] = BAD

    _, flagger_result = flagIsolated(data, field, flagger, group_window="1D", gap_window="2.1D", flag=BAD)

    assert flagger_result[field][slice(3, 6, 2)].all()

    data, flagger_result = flagIsolated(
        data, field, flagger_result,
        group_window="2D", gap_window="2.1D", continuation_range="1.1D", flag=BAD
    )
    assert flagger_result[field][[3, 5, 13, 14]].all()


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
    flagger = initFlagsLike(data)
    _, flagger_result = flagCrossStatistic(data, field, flagger, fields=fields, thresh=3, cross_stat=np.mean, flag=BAD)
    for field in fields:
        isflagged = flagger_result[field] > UNFLAGGED
        assert isflagged[characteristics["raise"]].all()


def test_flagManual(data, field):
    flagger = initFlagsLike(data)
    args = data, field, flagger
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
    _, fl = flagManual(*args, mdata=mdata, mflag="i do not exist", method="ontime", flag=BAD)
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
    data = dat(periods=200, peak_level=5, name='d1')[0]
    data['d2'] = dat(periods=200, peak_level=10, name='d2')[0]['d2']
    data['d3'] = dat(periods=200, peak_level=100, name='d3')[0]['d3']
    data['d4'] = 3 + 4 * data['d1']
    data['d5'] = 3 + 4 * data['d1']

    flagger = initFlagsLike(data)
    data_norm, flagger_norm = flagDriftFromNorm(
        data, 'dummy', flagger,
        ['d1', 'd2', 'd3'],
        segment_freq="200min",
        norm_spread=5,
        flag=BAD,
    )

    data_ref, flagger_ref = flagDriftFromReference(
        data, 'd1', flagger,
        ['d1', 'd2', 'd3'],
        segment_freq="3D",
        thresh=20,
        flag=BAD,
    )

    data_scale, flagger_scale = flagDriftFromScaledNorm(
        data, 'dummy', flagger,
        ['d1', 'd3'], ['d4', 'd5'],
        segment_freq="3D",
        thresh=20,
        norm_spread=5,
        flag=BAD,
    )
    assert all(flagger_norm['d3'] > UNFLAGGED)
    assert all(flagger_ref['d3'] > UNFLAGGED)
    assert all(flagger_scale['d3'] > UNFLAGGED)
