#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.flagger import Flagger, initFlagsLike
from saqc.constants import BAD, UNFLAGGED
from saqc.funcs.resampling import (
    linear,
    interpolate,
    shift,
    aggregate,
    mapToOriginal,
)


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


@pytest.mark.parametrize("reshaper", ["nshift", "fshift", "bshift", "nagg", "bagg", "fagg", "interpolation"])
def test_harmSingleVarIntermediateFlagging(data, reshaper):
    flagger = initFlagsLike(data)
    field = 'data'

    pre_data = data.copy()
    pre_flagger = flagger.copy()

    data, flagger = linear(data, field, flagger, freq="15min")

    # flag something bad
    flagger[data[field].index[3:4], field] = BAD
    data, flagger = mapToOriginal(data, field, flagger, method="inverse_" + reshaper)

    assert len(data[field]) == len(flagger[field])
    assert data[field].equals(pre_data[field])
    assert flagger[field].index.equals(pre_flagger[field].index)

    if 'agg' in reshaper:
        if reshaper == "nagg":
            start, end = 3, 7
        elif reshaper == "fagg":
            start, end = 3, 5
        elif reshaper == "bagg":
            start, end = 5, 7
        else:
            raise NotImplementedError('untested test case')

        assert all(flagger[field].iloc[start:end] > UNFLAGGED)
        assert all(~flagger[field].iloc[:start] == UNFLAGGED)
        assert all(~flagger[field].iloc[end:] == UNFLAGGED)

    elif 'shift' in reshaper:
        if reshaper == "nshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "fshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "bshift":
            exp = [False, False, False, False, False, True, False, False, False]
        else:
            raise NotImplementedError('untested test case')

        flagged = flagger[field] > UNFLAGGED
        assert all(flagged == exp)

    elif reshaper == 'interpolation':
        pytest.skip('no testcase for interpolation')

    else:
        raise NotImplementedError('untested test case')


@pytest.mark.parametrize(
    'params, expected',
    [
        (("nagg", "15Min"), pd.Series(data=[-87.5, -25.0, 0.0, 37.5, 50.0], index=pd.date_range("2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="15min"))),
        (("nagg", "30Min"), pd.Series(data=[-87.5, -25.0, 87.5], index=pd.date_range("2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="30min"))),
        (("bagg", "15Min"), pd.Series(data=[-50.0, -37.5, -37.5, 12.5, 37.5, 50.0], index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15min"))),
        (("bagg", "30Min"), pd.Series(data=[-50.0, -75.0, 50.0, 50.0], index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30min"))),
    ])
def test_harmSingleVarInterpolationAgg(data, params, expected):
    flagger = initFlagsLike(data)
    field = 'data'
    pre_data = data.copy()
    pre_flaggger = flagger.copy()
    method, freq = params

    data_harm, flagger_harm = aggregate(data, field, flagger, freq, value_func=np.sum, method=method)
    assert data_harm[field].equals(expected)

    data_deharm, flagger_deharm = mapToOriginal(data_harm, "data", flagger_harm, method="inverse_" + method)
    assert data_deharm[field].equals(pre_data[field])
    assert flagger_deharm[field].equals(pre_flaggger[field])


@pytest.mark.parametrize(
    'params, expected',
    [
        (("fshift", "15Min"), pd.Series(data=[np.nan, -37.5, -25.0, 0.0, 37.5, 50.0], index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"))),
        (("fshift", "30Min"), pd.Series(data=[np.nan, -37.5, 0.0, 50.0], index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"))),
        (("bshift", "15Min"), pd.Series(data=[-50.0, -37.5, -25.0, 12.5, 37.5, 50.0], index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"))),
        (("bshift", "30Min"), pd.Series(data=[-50.0, -37.5, 12.5, 50.0], index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"))),
        (("nshift", "15min"), pd.Series(data=[np.nan, -37.5, -25.0, 12.5, 37.5, 50.0], index=pd.date_range("2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"))),
        (("nshift", "30min"), pd.Series(data=[np.nan, -37.5, 12.5, 50.0], index=pd.date_range("2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"))),
    ])
def  test_harmSingleVarInterpolationShift(data, params, expected):
    flagger = initFlagsLike(data)
    field = 'data'
    pre_data = data.copy()
    pre_flagger = flagger.copy()
    method, freq = params

    data_harm, flagger_harm = shift(data, field, flagger, freq, method=method)
    assert data_harm[field].equals(expected)

    data_deharm, flagger_deharm = mapToOriginal(data_harm, "data", flagger_harm, method="inverse_" + method)
    assert data_deharm[field].equals(pre_data[field])
    assert flagger_deharm[field].equals(pre_flagger[field])


@pytest.mark.parametrize("method", ["time", "polynomial"])
def test_gridInterpolation(data, method):
    freq = "15min"
    field = 'data'
    data = data[field]
    data = (data * np.sin(data)).append(data.shift(1, "2h")).shift(1, "3s")
    data = dios.DictOfSeries(data)
    flagger = initFlagsLike(data)

    # we are just testing if the interpolation gets passed to the series without causing an error:
    interpolate(data, field, flagger, freq, method=method, downcast_interpolation=True)

    if method == "polynomial":
        interpolate(data, field, flagger, freq, order=2, method=method, downcast_interpolation=True)
        interpolate(data, field, flagger, freq, order=10, method=method, downcast_interpolation=True)


@pytest.mark.parametrize('func, kws', [
    ('linear', dict(to_drop=None)),
    ('shift', dict(method="nshift", to_drop=None)),
    ('interpolate', dict(method="spline")),
    ('aggregate', dict(value_func=np.nansum, method="nagg", to_drop=None)),
])
def test_wrapper(data, func, kws):
    # we are only testing, whether the wrappers do pass processing:
    field = 'data'
    freq = "15min"
    flagger = initFlagsLike(data)

    import saqc
    func = getattr(saqc.funcs, func)
    func(data, field, flagger, freq, **kws)

