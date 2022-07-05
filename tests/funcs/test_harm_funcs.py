#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

import dios
from saqc.constants import BAD, UNFLAGGED
from saqc.core import SaQC, initFlagsLike
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

    # GL-#352
    # make a History, otherwise nothing important is tested
    for c in flags.columns:
        flags[:, c] = BAD

    qc = SaQC(data, flags)

    qc = getattr(qc, func)(field, freq, **kws)

    # check minimal requirements
    checkDataFlagsInvariants(qc._data, qc._flags, field)
    assert qc.data[field].index.inferred_freq == freq


_SUPPORTED_METHODS = [
    "linear",
    "time",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
]


@pytest.mark.parametrize("method", _SUPPORTED_METHODS)
@pytest.mark.parametrize("fill_history", ["some", "all", "none"])
def test_gridInterpolation(data, method, fill_history):
    freq = "15T"
    field = "data"
    data = data[field]
    data = pd.concat([data * np.sin(data), data.shift(1, "2h")]).shift(1, "3s")
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)

    if fill_history == "none":
        pass

    if fill_history == "all":
        for c in flags.columns:
            flags[:, c] = BAD

    if fill_history == "some":
        for c in flags.columns:
            flags[::2, c] = UNFLAGGED

    qc = SaQC(data, flags)

    # we are just testing if the interpolation gets passed to the series without
    # causing an error:
    res = qc.interpolate(
        field,
        freq,
        method=method,
        downcast_interpolation=True,
    )

    if method == "polynomial":
        res = qc.interpolate(
            field,
            freq,
            order=2,
            method=method,
            downcast_interpolation=True,
        )
        res = qc.interpolate(
            field,
            freq,
            order=10,
            method=method,
            downcast_interpolation=True,
        )

    # check minimal requirements
    checkDataFlagsInvariants(res._data, res._flags, field, identical=False)
    assert res.data[field].index.inferred_freq == freq


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
    qc = SaQC(data, flags)

    qc = qc.copyField(field, field + "_interpolated")
    qc = qc.linear(field + "_interpolated", freq=freq)
    checkDataFlagsInvariants(
        qc._data, qc._flags, field + "_interpolated", identical=True
    )
    assert qc._data[field + "_interpolated"].index.inferred_freq == freq

    # flag something bad
    qc._flags[
        qc._data[field + "_interpolated"].index[3:4], field + "_interpolated"
    ] = BAD
    qc = qc.concatFlags(
        field + "_interpolated", method="inverse_" + reshaper, target=field
    )
    qc = qc.dropField(field + "_interpolated")

    assert len(qc.data[field]) == len(qc.flags[field])
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].index.equals(pre_flags[field].index)

    if "agg" in reshaper:
        if reshaper == "nagg":
            start, end = 3, 7
        elif reshaper == "fagg":
            start, end = 3, 5
        elif reshaper == "bagg":
            start, end = 5, 7
        else:
            raise NotImplementedError("untested test case")

        assert all(qc._flags[field].iloc[start:end] > UNFLAGGED)
        assert all(qc._flags[field].iloc[:start] == UNFLAGGED)
        assert all(qc._flags[field].iloc[end:] == UNFLAGGED)

    elif "shift" in reshaper:
        if reshaper == "nshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "fshift":
            exp = [False, False, False, False, True, False, False, False, False]
        elif reshaper == "bshift":
            exp = [False, False, False, False, False, True, False, False, False]
        else:
            raise NotImplementedError("untested test case")

        flagged = qc._flags[field] > UNFLAGGED
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

    qc = SaQC(data, flags)

    qc = qc.copyField("data", "data_harm")
    qc = qc.resample(h_field, freq, func=np.sum, method=method)

    checkDataFlagsInvariants(qc._data, qc._flags, h_field, identical=True)
    assert qc._data[h_field].index.freq == pd.Timedelta(freq)
    assert qc._data[h_field].equals(expected)

    qc = qc.concatFlags(h_field, target=field, method="inverse_" + method)
    qc = qc.dropField(h_field)
    checkDataFlagsInvariants(qc._data, qc._flags, field, identical=True)
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flaggger[field])


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

    qc = SaQC(data, flags)

    qc = qc.copyField("data", "data_harm")
    qc = qc.shift(h_field, freq, method=method)
    assert qc.data[h_field].equals(expected)
    checkDataFlagsInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.concatFlags(h_field, target=field, method="inverse_" + method)
    checkDataFlagsInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.dropField(h_field)
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flags[field])
