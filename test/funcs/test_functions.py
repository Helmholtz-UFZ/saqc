#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.funcs.functions import *
from test.common import initData, TESTFLAGGER




@pytest.fixture
def data():
    return initData(cols=1, start_date="2016-01-01", end_date="2018-12-31", freq="1D")


@pytest.fixture
def field(data):
    return data.columns[0]


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagRange(data, field, flagger):
    min, max = 10, 90
    flagger = flagger.initFlags(data)
    data, flagger = flagRange(data, field, flagger, min=min, max=max)
    flagged = flagger.isFlagged(field)
    expected = (data[field] < min) | (data[field] > max)
    assert (flagged == expected).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagSesonalRange(data, field, flagger):
    # prepare
    data.iloc[::2] = 0
    data.iloc[1::2] = 50
    nyears = len(data[field].index.year.unique())

    tests = [
        ({"min": 1, "max": 100, "startmonth": 7, "startday": 1, "endmonth": 8, "endday": 31,}, 31 * 2 * nyears // 2,),
        ({"min": 1, "max": 100, "startmonth": 12, "startday": 16, "endmonth": 1, "endday": 15,}, 31 * nyears // 2 + 1,),
    ]

    for test, expected in tests:
        flagger = flagger.initFlags(data)
        data, flagger = flagSesonalRange(data, field, flagger, **test)
        flagged = flagger.isFlagged(field)
        assert flagged.sum() == expected


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_clearFlags(data, field, flagger):
    flagger = flagger.initFlags(data)
    flags_orig = flagger.getFlags()
    flags_set = flagger.setFlags(field, flag=flagger.BAD).getFlags()
    _, flagger = clearFlags(data, field, flagger)
    flags_cleared = flagger.getFlags()
    assert (flags_orig != flags_set).all(None)
    assert (flags_orig == flags_cleared).all(None)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_forceFlags(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns
    flags_orig = flagger.setFlags(field).getFlags(field)
    _, flagger = forceFlags(data, field, flagger, flag=flagger.GOOD)
    flags_forced = flagger.getFlags(field)
    assert np.all(flags_orig != flags_forced)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagIsolated(data, flagger):
    field = data.columns[0]
    data.iloc[1:3, 0] = np.nan
    data.iloc[4:5, 0] = np.nan
    data.iloc[11:13, 0] = np.nan
    data.iloc[15:17, 0] = np.nan
    flagger = flagger.initFlags(data)
    s = data[field].iloc[5:6]
    flagger = flagger.setFlags(field, loc=s)

    _, flagger_result = flagIsolated(data, field, flagger, group_window="1D", gap_window="2.1D")

    assert flagger_result.isFlagged(field)[slice(3, 6, 2)].all()

    data, flagger_result = flagIsolated(
        data, field, flagger_result, group_window="2D", gap_window="2.1D", continuation_range="1.1D",
    )
    assert flagger_result.isFlagged(field)[[3, 5, 13, 14]].all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_flagCrossScoring(dat, flagger):
    data1, characteristics = dat(initial_level=0, final_level=0, out_val=0)
    data2, characteristics = dat(initial_level=0, final_level=0, out_val=10)
    field = "dummy"
    fields = ["data1", "data2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["data1", "data2"])
    flagger = flagger.initFlags(data)
    _, flagger_result = flagCrossScoring(data, field, flagger, fields=fields, thresh=3, cross_stat=np.mean)
    for field in fields:
        isflagged = flagger_result.isFlagged(field)
        assert isflagged[characteristics["raise"]].all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagManual(data, flagger):
    field = data.columns[0]
    flagger = flagger.initFlags(data)
    args = data, field, flagger
    dat = data[field]

    mdata = pd.Series("lala", index=dat.index)
    index_exp = mdata.iloc[[10, 33, 200, 500]].index
    mdata.iloc[[101, 133, 220, 506]] = "b"
    mdata.loc[index_exp] = "a"
    shrinked = mdata.loc[index_exp.union(mdata.iloc[[1, 2, 3, 4, 600, 601]].index)]

    kwargs_list = [
        dict(mdata=mdata, mflag="a", method="plain"),
        dict(mdata=mdata.to_list(), mflag="a", method="plain"),
        dict(mdata=mdata, mflag="a", method="ontime"),
        dict(mdata=shrinked, mflag="a", method="ontime"),
    ]

    for kw in kwargs_list:
        _, fl = flagManual(*args, **kw)
        isflagged = fl.isFlagged(field)
        assert isflagged[isflagged].index.equals(index_exp)

    # flag not exist in mdata
    _, fl = flagManual(*args, mdata=mdata, mflag="i do not exist", method="ontime")
    isflagged = fl.isFlagged(field)
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

    _, fl = flagManual(*args, mdata=mdata, mflag=1, method="right-open")
    isflagged = fl.isFlagged(field)
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
    _, fl = flagManual(*args, mdata=mdata, mflag=1, method="left-open")
    isflagged = fl.isFlagged(field)
    last = expected.index[0]
    assert isflagged[last] == expected[last]
    for curr in expected.index[1:]:
        expected_value = mdata[curr]
        # datetime slicing is inclusive !
        i = isflagged[last:curr].index[1:]
        chunk = isflagged.loc[i]
        assert (chunk == expected_value).all()
        last = curr

@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_flagDriftFromNormal(dat, flagger):
    data = dat(periods=200, peak_level=5, name='d1')[0]
    data['d2'] = dat(periods=200, peak_level=10, name='d2')[0]['d2']
    data['d3'] = dat(periods=200, peak_level=100, name='d3')[0]['d3']
    data['d4'] = 3 + 4 * data['d1']
    data['d5'] = 3 + 4 * data['d1']

    flagger = flagger.initFlags(data)
    data_norm, flagger_norm = flagDriftFromNorm(data, 'dummy', flagger, ['d1', 'd2', 'd3'], segment_freq="200min",
                                      norm_spread=5)

    data_ref, flagger_ref = flagDriftFromReference(data, 'd1', flagger, ['d1', 'd2', 'd3'], segment_freq="3D",
                                      thresh=20)

    data_scale, flagger_scale = flagDriftScale(data, 'dummy', flagger, ['d1', 'd3'], ['d4', 'd5'], segment_freq="3D",
                                                   thresh=20,  norm_spread=5)
    assert flagger_norm.isFlagged()['d3'].all()
    assert flagger_ref.isFlagged()['d3'].all()
    assert flagger_scale.isFlagged()['d3'].all()
