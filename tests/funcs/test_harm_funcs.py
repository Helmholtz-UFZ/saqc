#! /usr/bin/env python
# -*- coding: utf-8 -*-


# see test/functs/fixtures.py for global fixtures "course_..."
import pytest
import numpy as np
import pandas as pd
import dios

from tests.common import TESTFLAGGER

from saqc.funcs.resampling import (
    linear,
    interpolate,
    shift,
    aggregate,
    mapToOriginal,
)

RESHAPERS = ["nshift", "fshift", "bshift", "nagg", "bagg", "fagg", "interpolation"]

INTERPOLATIONS = ["time", "polynomial"]


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


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("reshaper", RESHAPERS)
def test_harmSingleVarIntermediateFlagging(data, flagger, reshaper):
    flagger = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flagger.getFlags()
    freq = "15min"
    assert len(data.columns) == 1
    field = data.columns[0]
    data, flagger = linear(data, "data", flagger, freq)
    # flag something bad
    flagger = flagger.setFlags("data", loc=data[field].index[3:4])
    data, flagger = mapToOriginal(data, "data", flagger, method="inverse_" + reshaper)
    d = data[field]
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
        assert (flagger.isFlagged().squeeze() == [False, False, False, False, True, False, False, False, False]).all()

    flags = flagger.getFlags()
    assert pre_data[field].equals(data[field])
    assert len(data[field]) == len(flags[field])
    assert (pre_flags[field].index == flags[field].index).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_harmSingleVarInterpolations(data, flagger):
    flagger = flagger.initFlags(data)
    field = data.columns[0]
    pre_data = data[field]
    pre_flags = flagger.getFlags(field)
    tests = [
        (
            "nagg",
            "15Min",
            pd.Series(
                data=[-87.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range("2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="15min"),
            ),
        ),
        (
            "nagg",
            "30Min",
            pd.Series(
                data=[-87.5, -25.0, 87.5],
                index=pd.date_range("2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="30min"),
            ),
        ),
        (
            "bagg",
            "15Min",
            pd.Series(
                data=[-50.0, -37.5, -37.5, 12.5, 37.5, 50.0],
                index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15min"),
            ),
        ),
        (
            "bagg",
            "30Min",
            pd.Series(
                data=[-50.0, -75.0, 50.0, 50.0],
                index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30min"),
            ),
        ),
    ]

    for interpolation, freq, expected in tests:
        data_harm, flagger_harm = aggregate(
            data, field, flagger, freq, value_func=np.sum, method=interpolation
        )
        assert data_harm[field].equals(expected)
        data_deharm, flagger_deharm = mapToOriginal(
            data_harm, "data", flagger_harm, method="inverse_" + interpolation
        )
        assert data_deharm[field].equals(pre_data)
        assert flagger_deharm.getFlags([field]).squeeze().equals(pre_flags)

    tests = [
        (
            "fshift",
            "15Min",
            pd.Series(
                data=[np.nan, -37.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"),
            ),
        ),
        (
            "fshift",
            "30Min",
            pd.Series(
                data=[np.nan, -37.5, 0.0, 50.0],
                index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"),
            ),
        ),
        (
            "bshift",
            "15Min",
            pd.Series(
                data=[-50.0, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"),
            ),
        ),
        (
            "bshift",
            "30Min",
            pd.Series(
                data=[-50.0, -37.5, 12.5, 50.0],
                index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"),
            ),
        ),
        (
            "nshift",
            "15min",
            pd.Series(
                data=[np.nan, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"),
            ),
        ),
        (
            "nshift",
            "30min",
            pd.Series(
                data=[np.nan, -37.5, 12.5, 50.0],
                index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"),
            ),
        ),
    ]

    for interpolation, freq, expected in tests:
        data_harm, flagger_harm = shift(data, field, flagger, freq, method=interpolation)
        assert data_harm[field].equals(expected)
        data_deharm, flagger_deharm = mapToOriginal(
            data_harm, "data", flagger_harm, method="inverse_" + interpolation
        )
        assert data_deharm[field].equals(pre_data)
        assert flagger_deharm.getFlags([field]).squeeze().equals(pre_flags)


@pytest.mark.parametrize("method", INTERPOLATIONS)
def test_gridInterpolation(data, method):
    freq = "15min"
    data = data.squeeze()
    field = data.name
    data = (data * np.sin(data)).append(data.shift(1, "2h")).shift(1, "3s")
    data = dios.DictOfSeries(data)
    flagger = TESTFLAGGER[0].initFlags(data)

    # we are just testing if the interpolation gets passed to the series without causing an error:

    interpolate(data, field, flagger, freq, method=method, downcast_interpolation=True)
    if method == "polynomial":
        interpolate(data, field, flagger, freq, order=2, method=method, downcast_interpolation=True)
        interpolate(data, field, flagger, freq, order=10, method=method, downcast_interpolation=True)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_wrapper(data, flagger):
    # we are only testing, whether the wrappers do pass processing:
    field = data.columns[0]
    freq = "15min"
    flagger = flagger.initFlags(data)

    linear(data, field, flagger, freq, to_drop=None)
    aggregate(data, field, flagger, freq, value_func=np.nansum, method="nagg", to_drop=None)
    shift(data, field, flagger, freq, method="nshift", to_drop=None)
    interpolate(data, field, flagger, freq, method="spline")
