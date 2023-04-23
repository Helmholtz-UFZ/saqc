#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from saqc import BAD, UNFLAGGED, SaQC
from saqc.constants import FILTER_NONE
from saqc.core import DictOfSeries, Flags
from tests.common import initData


@pytest.fixture
def data():
    return initData()


def test_emptyData():
    # test that things do not break with empty data sets
    saqc = SaQC(data=pd.DataFrame({"x": [], "y": []}))

    saqc.flagGeneric("x", func=lambda x: x < 0)
    assert saqc.data.empty
    assert saqc.flags.empty

    saqc = saqc.processGeneric(field="x", target="y", func=lambda x: x + 2)
    assert saqc.data.empty
    assert saqc.flags.empty


@pytest.mark.parametrize(
    "targets, func",
    [
        (["tmp"], lambda x, y: pd.Series(True, index=x.index.union(y.index))),
        (
            ["tmp1", "tmp2"],
            lambda x, y: [pd.Series(True, index=x.index.union(y.index))] * 2,
        ),
    ],
)
def test_writeTargetFlagGeneric(data, targets, func):
    saqc = SaQC(data=data)
    saqc = saqc.flagGeneric(field=data.columns, target=targets, func=func, flag=BAD)
    for target in targets:
        assert saqc._flags.history[target].hist.iloc[0].tolist() == [BAD]


@pytest.mark.parametrize(
    "fields, func",
    [
        (["var1"], lambda x: pd.Series(True, index=x.index)),
        (
            ["var1", "var2"],
            lambda x, y: [pd.Series(True, index=x.index.union(y.index))] * 2,
        ),
    ],
)
def test_overwriteFieldFlagGeneric(data, fields, func):
    flag = 12

    saqc = SaQC(
        data=data.copy(),
        flags=Flags(
            {
                k: pd.Series(data[k] % 2, index=data[k].index).replace(
                    {0: UNFLAGGED, 1: 127}
                )
                for k in data.columns
            }
        ),
    )

    res = saqc.flagGeneric(field=fields, func=func, flag=flag, dfilter=FILTER_NONE)
    for field in fields:
        histcol1 = res._flags.history[field].hist[1]
        assert (histcol1 == flag).all()
        assert (data[field] == res.data[field]).all(axis=None)
        assert res._flags.history[field].meta[0] == {}


@pytest.mark.parametrize(
    "data, targets, func, expected_data",
    [
        (
            DictOfSeries(dict(a=pd.Series([1.0, 2.0]), b=pd.Series([10.0, 20.0]))),
            ["t1"],
            lambda x, y: x + y,
            DictOfSeries(
                dict(
                    a=pd.Series([1.0, 2.0]),
                    b=pd.Series([10.0, 20.0]),
                    t1=pd.Series([11.0, 22.0]),
                )
            ),
        ),
        (
            DictOfSeries(dict(a=pd.Series([1.0, 2.0]), b=pd.Series([10.0, 20.0]))),
            ["t1", "t2"],
            lambda x, y: (x + y, y * 2),
            DictOfSeries(
                dict(
                    a=pd.Series([1.0, 2.0]),
                    b=pd.Series([10.0, 20.0]),
                    t1=pd.Series([11.0, 22.0]),
                    t2=pd.Series([20.0, 40.0]),
                )
            ),
        ),
    ],
)
def test_writeTargetProcGeneric(data, targets, func, expected_data):
    fields = data.columns.tolist()
    dfilter = 128

    saqc = SaQC(
        data=data,
        flags=Flags({k: pd.Series(127.0, index=data[k].index) for k in data.columns}),
    )
    res = saqc.processGeneric(
        field=fields,
        target=targets,
        func=func,
        dfilter=dfilter,
        label="generic",
    )
    assert expected_data == res.data
    # check that new histories where created
    for target in targets:
        assert res._flags.history[target].hist.iloc[0].isna().all()


@pytest.mark.parametrize(
    "data, fields, func, expected_data",
    [
        (
            DictOfSeries(dict(a=pd.Series([1.0, 2.0]), b=pd.Series([10.0, 20.0]))),
            ["a"],
            lambda x: x * 2,
            DictOfSeries(dict(a=pd.Series([2.0, 4.0]), b=pd.Series([10.0, 20.0]))),
        ),
        (
            DictOfSeries(dict(a=pd.Series([1.0, 2.0]), b=pd.Series([10.0, 20.0]))),
            ["a", "b"],
            lambda x, y: (x + y, y * 2),
            DictOfSeries(dict(a=pd.Series([11.0, 22.0]), b=pd.Series([20.0, 40.0]))),
        ),
    ],
)
def test_overwriteFieldProcGeneric(data, fields, func, expected_data):
    dfilter = 128

    saqc = SaQC(
        data=data,
        flags=Flags({k: pd.Series(127.0, index=data[k].index) for k in data.columns}),
    )

    res = saqc.processGeneric(field=fields, func=func, dfilter=dfilter, label="generic")
    assert expected_data == res.data
    # check that the histories got appended
    for field in fields:
        assert (res._flags.history[field].hist[0] == 127.0).all()
        assert res._flags.history[field].hist[1].isna().all()
        assert res._flags.history[field].meta[0] == {}


def test_label():
    dat = pd.DataFrame(
        {"data1": [1, 1, 5, 2, 1], "data2": [1, 1, 2, 3, 4], "data3": [1, 1, 2, 3, 4]},
        index=pd.date_range("2000", "2005", periods=5),
    )

    qc = SaQC(dat)
    qc = qc.flagRange("data1", max=4, label="out of range")
    qc = qc.flagRange("data1", max=0, label="out of range2")
    qc = qc.flagGeneric(
        ["data1", "data3"],
        target="data2",
        func=lambda x, y: isflagged(x, "out of range") | isflagged(y),  # noqa
    )
    assert list((qc.flags["data2"] > 0).values) == [False, False, True, False, False]


@pytest.mark.parametrize(
    "kwargs, got, expected",
    [
        (
            {
                "lower": 0,
            },
            [-9, -2, 1, 2, 9],
            [0, 0, 1, 2, 9],
        ),
        ({"upper": 3}, [-9, -2, 1, 2, 9], [-9, -2, 1, 2, 3]),
        ({"lower": -6, "upper": 3}, [-9, -2, 1, 2, 9], [-6, -2, 1, 2, 3]),
    ],
)
def test_processGenericClip(kwargs, got, expected):
    field = "data"
    got = pd.DataFrame(
        got, columns=[field], index=pd.date_range("2020-06-30", periods=len(got))
    )
    expected = pd.DataFrame(
        expected,
        columns=[field],
        index=pd.date_range("2020-06-30", periods=len(expected)),
    )
    qc = SaQC(got).processGeneric(field, func=lambda x: clip(x, **kwargs))
    assert (qc._data[field] == expected[field]).all()
