#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import pandas as pd

from dios import dios
from test.common import TESTFLAGGER, initData

from saqc.funcs.harm_functions import (
    harm_harmonize,
    harm_deharmonize,
    _interpolateGrid,
    _insertGrid,
    _outsortCrap,
    harm_linear2Grid,
    harm_interpolate2Grid,
    harm_shift2Grid,
    harm_aggregate2Grid,
    harm_deharm
)
from saqc.funcs.proc_functions import ORIGINAL_SUFFIX


RESHAPERS = ["nshift", "fshift", "bshift", "nagg", "bagg", "fagg"]

COFLAGGING = [False]

SETSHIFTCOMMENT = [False, True]

INTERPOLATIONS2 = ["time", "polynomial"]


@pytest.fixture
def data():
    index = pd.date_range(start="1.1.2011 00:00:00", end="1.1.2011 01:00:00", freq="15min")
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp("2011-01-01 00:30:00"))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name="data")
    # good to have some nan
    dat[-3] = np.nan
    data = dios.DictOfSeries(dat)
    return data


@pytest.fixture
def multi_data():
    index = pd.date_range(start="1.1.2011 00:00:00", end="1.1.2011 01:00:00", freq="15min")
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp("2011-01-01 00:30:00"))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name="data")
    # good to have some nan
    dat[-3] = np.nan
    data = dat.to_frame()
    data.index = data.index.shift(1, "2min")
    dat2 = data.copy()
    dat2.index = dat2.index.shift(1, "17min")
    dat2.rename(columns={"data": "data2"}, inplace=True)
    dat3 = data.copy()
    dat3.index = dat3.index.shift(1, "1h")
    dat3.rename(columns={"data": "data3"}, inplace=True)
    dat3.drop(dat3.index[2:-2], inplace=True)
    # merge
    data = pd.merge(data, dat2, how="outer", left_index=True, right_index=True)
    data = pd.merge(data, dat3, how="outer", left_index=True, right_index=True)
    return dios.DictOfSeries(data)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("reshaper", RESHAPERS)
@pytest.mark.parametrize("co_flagging", COFLAGGING)
def test_harmSingleVarIntermediateFlagging(data, flagger, reshaper, co_flagging):
    flagger = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flagger.getFlags()
    freq = "15min"

    assert len(data.columns) == 1
    field = data.columns[0]

    data, flagger = harm_linear2Grid(data, 'data', flagger, freq)
    # flag something bad
    flagger = flagger.setFlags("data", loc=data[field].index[3:4])
    data, flagger = harm_deharm(data, "data", flagger, method='inverse_' + reshaper)
    d = data[field + ORIGINAL_SUFFIX]

    if reshaper == "nagg":
        assert flagger.isFlagged(loc=d.index[3:7]).squeeze().all()
        assert (~flagger.isFlagged(loc=d.index[0:3]).squeeze()).all()
        assert (~flagger.isFlagged(loc=d.index[7:]).squeeze()).all()
    if reshaper == "nshift":
        assert (flagger.isFlagged().squeeze() == [False, False, False, False, True, False, False, False, False]).all()
    if reshaper == "bagg":
        assert flagger.isFlagged(loc=d.index[5:7]).squeeze().all()
        assert (~flagger.isFlagged(loc=d.index[0:5]).squeeze()).all()
        assert (~flagger.isFlagged(loc=d.index[7:]).squeeze()).all()
    if reshaper == "bshift":
        assert (flagger.isFlagged().squeeze() == [False, False, False, False, False, True, False, False, False]).all()
    if reshaper == "fagg":
        assert flagger.isFlagged(loc=d.index[3:5]).squeeze().all()
        assert (~flagger.isFlagged(loc=d.index[0:3]).squeeze()).all()
        assert (~flagger.isFlagged(loc=d.index[5:]).squeeze()).all()
    if reshaper == "fshift":
        assert (flagger.isFlagged().squeeze() == [False, False, False, False, True, False, False, False, False]
            ).all()

    flags = flagger.getFlags()
    assert pre_data[field].equals(data[field + ORIGINAL_SUFFIX])
    assert len(data[field + ORIGINAL_SUFFIX]) == len(flags[field + ORIGINAL_SUFFIX])
    assert (pre_flags[field].index == flags[field + ORIGINAL_SUFFIX].index).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_harmSingleVarInterpolations(data, flagger):
    flagger = flagger.initFlags(data)

    field = data.columns[0]

    tests = [
        ("nagg", "15Min", pd.Series(data=[-87.5, -25.0, 0.0, 37.5, 50.0],
                                    index=pd.date_range('2011-01-01 00:00:00',
                                                        '2011-01-01 01:00:00',
                                                        freq='15min'))),
        ("nagg", "30Min", pd.Series(data=[-87.5, -25.0, 87.5],
                                    index=pd.date_range('2011-01-01 00:00:00',
                                                        '2011-01-01 01:00:00',
                                                        freq='30min'))),
        ("bagg", "15Min", pd.Series(data=[-50.0, -37.5, -37.5, 12.5, 37.5, 50.0],
                                    index=pd.date_range('2010-12-31 23:45:00',
                                                        '2011-01-01 01:00:00',
                                                        freq='15min'))),
        ("bagg", "30Min", pd.Series(data=[-50.0, -75.0, 50.0, 50.0],
                                    index=pd.date_range('2010-12-31 23:30:00',
                                                        '2011-01-01 01:00:00',
                                                        freq='30min'))),
    ]

    for interpolation, freq, expected in tests:
        data_harm, _ = harm_aggregate2Grid(data, field, flagger, freq, value_func=np.sum, method=interpolation)
        assert data_harm[field].equals(expected)

    #tests = [
    #    ("fshift", "15Min", [np.nan, -37.5, -25.0, 0.0, 37.5, 50.0]),
    #    ("fshift", "30Min", [np.nan, -37.5, 0.0, 50.0]),
    #    ("bshift", "15Min", [-50.0, -37.5, -25.0, 12.5, 37.5, 50.0]),
    #    ("bshift", "30Min", [-50.0, -37.5, 12.5, 50.0]),
    #    ("nshift", "15min", [np.nan, -37.5, -25.0, 12.5, 37.5, 50.0]),
    #    ("nshift", "30min", [np.nan, -37.5, 12.5, 50.0])]

    #for interpolation, freq, expected in tests:
    #    data_harm, _ = harm_aggregate2Grid(data, field, flagger, freq, value_func=np.sum, method=interpolation)
    #    harm_start = data[field].index[0].floor(freq=freq)
    #    harm_end = data[field].index[-1].ceil(freq=freq)
    #    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    #    expected = pd.Series(expected, index=test_index)
    #    assert data_harm[field].equals(expected)

    #data_deharm, flagger_deharm = harm_deharmonize(data, "data", flagger, co_flagging=True)


    #flags = flagger.getFlags()
    #flags_deharm = flagger_deharm.getFlags()

    #assert data[field].equals(data[field])
    #assert len(data_deharm[field]) == len(flags[field])
    #assert (flags[field].index == flags_deharm[field].index).all()


@pytest.mark.parametrize("method", INTERPOLATIONS2)
def test_gridInterpolation(data, method):
    freq = "15min"
    data = data.squeeze()
    field = data.name
    data = (data * np.sin(data)).append(data.shift(1, "2h")).shift(1, "3s")
    data = dios.DictOfSeries(data)
    flagger = TESTFLAGGER[0].initFlags(data)

    # we are just testing if the interpolation gets passed to the series without causing an error:

    harm_interpolate2Grid(data, field, flagger, freq, method=method, downcast_interpolation=True)
    if method == "polynomial":
        harm_interpolate2Grid(data, field, flagger, freq, order=2, method=method, downcast_interpolation=True)
        harm_interpolate2Grid(data, field, flagger, freq, order=10, method=method, downcast_interpolation=True)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_outsortCrap(data, flagger):

    field = data.columns[0]
    flagger = flagger.initFlags(data)
    flagger = flagger.setFlags(field, loc=data[field].index[5:7])

    drop_index = data[field].index[5:7]
    d, *_ = _outsortCrap(data[field], field, flagger, drop_flags=flagger.BAD)
    assert drop_index.difference(d.index).equals(drop_index)

    flagger = flagger.setFlags(field, loc=data[field].index[0:1], flag=flagger.GOOD)
    drop_index = drop_index.insert(-1, data[field].index[0])
    d, *_ = _outsortCrap(data[field], field, flagger, drop_flags=[flagger.BAD, flagger.GOOD],)
    assert drop_index.sort_values().difference(d.index).equals(drop_index.sort_values())


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_wrapper(data, flagger):
    # we are only testing, whether the wrappers do pass processing:
    field = data.columns[0]
    freq = "15min"
    flagger = flagger.initFlags(data)

    harm_linear2Grid(data, field, flagger, freq, drop_flags=None)
    harm_aggregate2Grid(data, field, flagger, freq, value_func=np.nansum, method="nagg", drop_flags=None)
    harm_shift2Grid(data, field, flagger, freq, method="nshift", drop_flags=None)
    harm_interpolate2Grid(data, field, flagger, freq, method="spline")
