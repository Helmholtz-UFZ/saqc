#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.core import initFlagsLike, Flags
from saqc.constants import BAD, UNFLAGGED
from saqc.funcs.resampling import linear, interpolate, shift, concatFlags, resample
from saqc.funcs.tools import copyField, dropField
from tests.common import checkDataFlagsInvariants


@pytest.fixture
def data():
    index = pd.date_range(
        start="1.1.2011 00:00:00", end="1.1.2011 01:00:00", freq="15min"
    )
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


@pytest.mark.parametrize(
    "func, kws",
    [
        ("linear", dict()),
        ("shift", dict(method="nshift")),
        ("interpolate", dict(method="spline")),
        ("resample", dict(func=np.nansum, method="nagg")),
    ],
)
def test_wrapper(data, func, kws):
    field = "data"
    freq = "15T"
    flags = initFlagsLike(data)

    import saqc

    func = getattr(saqc.funcs, func)
    data, flags = func(data, field, flags, freq, **kws)

    # check minimal requirements
    checkDataFlagsInvariants(data, flags, field)
    assert data[field].index.inferred_freq == freq


@pytest.mark.parametrize("method", ["time", "polynomial"])
def test_gridInterpolation(data, method):
    freq = "15T"
    field = "data"
    data = data[field]
    data = (data * np.sin(data)).append(data.shift(1, "2h")).shift(1, "3s")
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)

    # we are just testing if the interpolation gets passed to the series without
    # causing an error:
    res = interpolate(
        data.copy(),
        field,
        flags.copy(),
        freq,
        method=method,
        downcast_interpolation=True,
    )

    if method == "polynomial":
        res = interpolate(
            data.copy(),
            field,
            flags.copy(),
            freq,
            order=2,
            method=method,
            downcast_interpolation=True,
        )
        res = interpolate(
            data.copy(),
            field,
            flags.copy(),
            freq,
            order=10,
            method=method,
            downcast_interpolation=True,
        )

    # check minimal requirements
    rdata, rflags = res
    checkDataFlagsInvariants(rdata, rflags, field, identical=False)
    assert rdata[field].index.inferred_freq == freq


@pytest.mark.parametrize(
    "func, kws",
    [
        ("linear", dict()),
        ("shift", dict(method="nshift")),
        ("interpolate", dict(method="spline")),
        ("aggregate", dict(value_func=np.nansum, method="nagg")),
    ],
)
def test_flagsSurviveReshaping(func, kws):
    """
    flagging -> reshaping -> test (flags also was reshaped correctly)
    """
    pass


def test_flagsSurviveInverseReshaping():
    """
    inverse reshaping -> flagging -> test (flags also was reshaped correctly)"""
    pass


def test_flagsSurviveBackprojection():
    """
    flagging -> reshaping -> inverse reshaping -> test (flags == original-flags)
    """
    pass


@pytest.mark.parametrize(
    "reshaper", ["nshift", "fshift", "bshift", "nagg", "bagg", "fagg", "interpolation"]
)
def test_harmSingleVarIntermediateFlagging(data, reshaper):
    flags = initFlagsLike(data)
    field = "data"
    freq = "15T"

    pre_data = data.copy()
    pre_flags = flags.copy()
    data, flags = copyField(data, field, flags, field + "_interpolated")
    data, flags = linear(data, field + "_interpolated", flags, freq=freq)
    checkDataFlagsInvariants(data, flags, field + "_interpolated", identical=True)
    assert data[field + "_interpolated"].index.inferred_freq == freq

    # flag something bad
    flags[data[field + "_interpolated"].index[3:4], field + "_interpolated"] = BAD
    data, flags = concatFlags(
        data, field + "_interpolated", flags, method="inverse_" + reshaper, target=field
    )
    data, flags = dropField(data, field + "_interpolated", flags)

    assert len(data[field]) == len(flags[field])
    assert data[field].equals(pre_data[field])
    assert flags[field].index.equals(pre_flags[field].index)

    if "agg" in reshaper:
        if reshaper == "nagg":
            start, end = 3, 7
        elif reshaper == "fagg":
            start, end = 3, 5
        elif reshaper == "bagg":
            start, end = 5, 7
        else:
            raise NotImplementedError("untested test case")

        assert all(flags[field].iloc[start:end] > UNFLAGGED)
        assert all(flags[field].iloc[:start] == UNFLAGGED)
        assert all(flags[field].iloc[end:] == UNFLAGGED)

    elif "shift" in reshaper:
        if reshaper == "nshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "fshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "bshift":
            exp = [False, False, False, False, False, True, False, False, False]
        else:
            raise NotImplementedError("untested test case")

        flagged = flags[field] > UNFLAGGED
        assert all(flagged == exp)

    elif reshaper == "interpolation":
        pytest.skip("no testcase for interpolation")

    else:
        raise NotImplementedError("untested test case")


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            ("nagg", "15Min"),
            pd.Series(
                data=[-87.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range(
                    "2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="15min"
                ),
            ),
        ),
        (
            ("nagg", "30Min"),
            pd.Series(
                data=[-87.5, -25.0, 87.5],
                index=pd.date_range(
                    "2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="30min"
                ),
            ),
        ),
        (
            ("bagg", "15Min"),
            pd.Series(
                data=[-50.0, -37.5, -37.5, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15min"
                ),
            ),
        ),
        (
            ("bagg", "30Min"),
            pd.Series(
                data=[-50.0, -75.0, 50.0, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30min"
                ),
            ),
        ),
    ],
)
def test_harmSingleVarInterpolationAgg(data, params, expected):
    flags = initFlagsLike(data)
    field = "data"
    h_field = "data_harm"

    pre_data = data.copy()
    pre_flaggger = flags.copy()
    method, freq = params

    data_harm, flags_harm = copyField(data, "data", flags, "data_harm")
    data_harm, flags_harm = resample(
        data_harm, h_field, flags_harm, freq, func=np.sum, method=method
    )
    checkDataFlagsInvariants(data_harm, flags_harm, h_field, identical=True)
    assert data_harm[h_field].index.freq == pd.Timedelta(freq)
    assert data_harm[h_field].equals(expected)

    data_deharm, flags_deharm = concatFlags(
        data_harm, h_field, flags_harm, target=field, method="inverse_" + method
    )
    data_deharm, flags_deharm = dropField(data_deharm, h_field, flags_deharm)
    checkDataFlagsInvariants(data_deharm, flags_deharm, field, identical=True)
    assert data_deharm[field].equals(pre_data[field])
    assert flags_deharm[field].equals(pre_flaggger[field])


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            ("bshift", "15Min"),
            pd.Series(
                data=[-50.0, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            ("fshift", "15Min"),
            pd.Series(
                data=[np.nan, -37.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            ("nshift", "15min"),
            pd.Series(
                data=[np.nan, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            ("bshift", "30Min"),
            pd.Series(
                data=[-50.0, -37.5, 12.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
        (
            ("fshift", "30Min"),
            pd.Series(
                data=[np.nan, -37.5, 0.0, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
        (
            ("nshift", "30min"),
            pd.Series(
                data=[np.nan, -37.5, 12.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
    ],
)
def test_harmSingleVarInterpolationShift(data, params, expected):
    flags = initFlagsLike(data)
    field = "data"
    h_field = "data_harm"
    pre_data = data.copy()
    pre_flags = flags.copy()
    method, freq = params

    data_harm, flags_harm = copyField(data, "data", flags, "data_harm")
    data_harm, flags_harm = shift(data_harm, h_field, flags_harm, freq, method=method)
    assert data_harm[h_field].equals(expected)
    checkDataFlagsInvariants(data_harm, flags_harm, field, identical=True)

    data_deharm, flags_deharm = concatFlags(
        data_harm, h_field, flags_harm, target=field, method="inverse_" + method
    )
    checkDataFlagsInvariants(data_deharm, flags_deharm, field, identical=True)

    data_deharm, flags_deharm = dropField(data_deharm, h_field, flags_deharm)
    assert data_deharm[field].equals(pre_data[field])
    assert flags_deharm[field].equals(pre_flags[field])
