#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from saqc import SaQC
from saqc.core import DictOfSeries, initFlagsLike
from tests.common import checkInvariants


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
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index)
    # good to have some nan
    dat[-3] = np.nan
    data = DictOfSeries(data=dat)
    return data


def test_flagsSurviveReshaping():
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
    "method, freq, expected",
    [
        (
            "nagg",
            "15Min",
            pd.Series(
                data=[-87.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range(
                    "2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="15min"
                ),
            ),
        ),
        (
            "nagg",
            "30Min",
            pd.Series(
                data=[-87.5, -25.0, 87.5],
                index=pd.date_range(
                    "2011-01-01 00:00:00", "2011-01-01 01:00:00", freq="30min"
                ),
            ),
        ),
        (
            "bagg",
            "15Min",
            pd.Series(
                data=[-50.0, -37.5, -37.5, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15min"
                ),
            ),
        ),
        (
            "bagg",
            "30Min",
            pd.Series(
                data=[-50.0, -75.0, 50.0, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30min"
                ),
            ),
        ),
    ],
)
def test_resampleAggregateInvert(data, method, freq, expected):
    flags = initFlagsLike(data)
    field = "data"
    field_aggregated = "data_aggregated"

    pre_data = data.copy()
    pre_flaggger = flags.copy()

    qc = SaQC(data, flags)

    qc = qc.copyField(field, field_aggregated)

    qc = qc.resample(field_aggregated, freq, func=np.sum, method=method)
    assert qc._data[field_aggregated].index.freq == pd.Timedelta(freq)
    assert qc._data[field_aggregated].equals(expected)
    assert qc._flags.history[field_aggregated].meta[-1]["func"] == "resample"
    checkInvariants(qc._data, qc._flags, field_aggregated, identical=True)

    qc = qc.concatFlags(field_aggregated, target=field, method="inverse_" + method)
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flaggger[field])
    checkInvariants(qc._data, qc._flags, field, identical=True)


@pytest.mark.parametrize(
    "method, freq, expected",
    [
        (
            "linear",
            "15Min",
            pd.Series(
                data=[np.nan, -37.5, -25, 6.25, 37.50, 50],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            "time",
            "30Min",
            pd.Series(
                data=[np.nan, -37.5, 6.25, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
        (
            "pad",
            "30Min",
            pd.Series(
                data=[np.nan, -37.5, 0, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
    ],
)
def test_alignInterpolateInvert(data, method, freq, expected):
    flags = initFlagsLike(data)

    field = "data"
    field_aligned = "data_aligned"

    pre_data = data.copy()
    pre_flags = flags.copy()

    qc = SaQC(data, flags)

    qc = qc.copyField(field, field_aligned)
    qc = qc.align(field_aligned, freq, method=method)

    assert qc.data[field_aligned].equals(expected)
    checkInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.concatFlags(field_aligned, target=field, method="inverse_interpolation")
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flags[field])
    checkInvariants(qc._data, qc._flags, field, identical=True)


@pytest.mark.parametrize(
    "method, freq, expected",
    [
        (
            "bshift",
            "15Min",
            pd.Series(
                data=[-50.0, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            "fshift",
            "15Min",
            pd.Series(
                data=[np.nan, -37.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            "nshift",
            "15min",
            pd.Series(
                data=[np.nan, -37.5, -25.0, 12.5, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15Min"
                ),
            ),
        ),
        (
            "bshift",
            "30Min",
            pd.Series(
                data=[-50.0, -37.5, 12.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
        (
            "fshift",
            "30Min",
            pd.Series(
                data=[np.nan, -37.5, 0.0, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
        (
            "nshift",
            "30min",
            pd.Series(
                data=[np.nan, -37.5, 12.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30Min"
                ),
            ),
        ),
    ],
)
def test_alignShiftInvert(data, method, freq, expected):
    flags = initFlagsLike(data)

    field = "data"
    field_aligned = "data_aligned"

    pre_data = data.copy()
    pre_flags = flags.copy()

    qc = SaQC(data, flags)

    qc = qc.copyField(field, field_aligned)
    qc = qc.align(field_aligned, freq, method=method)
    meta = qc._flags.history[field_aligned].meta[-1]

    assert qc.data[field_aligned].equals(expected)
    assert (meta["func"], meta["kwargs"]["method"]) == ("align", method)
    checkInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.concatFlags(field_aligned, target=field, method="inverse_" + method)
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flags[field])
    checkInvariants(qc._data, qc._flags, field, identical=True)


@pytest.mark.parametrize(
    "overwrite, expected_col0, expected_col1",
    [
        (
            True,
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 255, 255],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 255, np.nan, 255, 255],
        ),
        (
            False,
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 255, 255],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 255, np.nan, np.nan, np.nan],
        ),
    ],
)
def test_concatFlags(data, overwrite, expected_col0, expected_col1):
    qc = SaQC(data)

    qc = qc.flagRange(field="data", max=20)

    # branch out to another variable
    qc = qc.flagRange(field="data", target="data_", max=3)

    # bring the flags back again - overwrite
    qc_concat = qc.concatFlags(
        "data_", target="data", overwrite=overwrite, squeeze=True
    )
    hist_concat = qc_concat._flags.history["data"].hist.astype(float)
    meta_concat = qc_concat._flags.history["data"].meta
    assert hist_concat[0].equals(pd.Series(expected_col0, index=data["data"].index))
    assert hist_concat[1].equals(pd.Series(expected_col1, index=data["data"].index))
    assert meta_concat[-1]["func"] == "concatFlags"


@pytest.mark.parametrize(
    "method, inversion_method, freq",
    [
        ("linear", "inverse_interpolation", "15min"),
        ("bshift", "inverse_bshift", "15Min"),
        ("fshift", "inverse_fshift", "15Min"),
        ("nshift", "inverse_nshift", "15min"),
        ("pad", "inverse_interpolation", "15min"),
    ],
)
def test_alignAutoInvert(data, method, inversion_method, freq):
    flags = initFlagsLike(data)
    field = data.columns[0]
    field_aligned = f"{field}_aligned"

    qc = SaQC(data, flags)
    qc = qc.align(field=field, target=field_aligned, method=method, freq=freq)
    qc = qc.flagDummy(field=field_aligned)
    qc_expected = qc.concatFlags(
        field=field_aligned, target=field, method=inversion_method
    )
    qc_got = qc.concatFlags(field=field_aligned, target=field, method="auto")

    _assertEqual(qc_expected, qc_got)


def test_alignMultiAutoInvert(data):
    flags = initFlagsLike(data)
    field = data.columns[0]
    field_aligned = f"{field}_aligned"

    qc = SaQC(data, flags)
    qc = qc.align(field=field, target=field_aligned, method="fshift", freq="30Min")
    qc = qc.align(field=field_aligned, method="time", freq="10Min")
    qc = qc.flagDummy(field=field_aligned)

    # resolve the last alignment operation
    _assertEqual(
        qc.concatFlags(field=field_aligned, target=field, method="auto"),
        qc.concatFlags(
            field=field_aligned, target=field, method="inverse_interpolation"
        ),
    )
    # resolve the first alignment operation
    _assertEqual(
        (
            qc.concatFlags(field=field_aligned, method="auto").concatFlags(
                field=field_aligned, target=field, method="auto"
            )
        ),
        (
            qc.concatFlags(
                field=field_aligned, method="inverse_interpolation"
            ).concatFlags(field=field_aligned, target=field, method="inverse_fshift")
        ),
    )


def _assertEqual(left: SaQC, right: SaQC):
    for field in left.data.columns:
        assert left._data[field].equals(right._data[field])
        assert left._flags[field].equals(right._flags[field])
        assert left._flags.history[field].hist.equals(right._flags.history[field].hist)
        assert left._flags.history[field].meta == right._flags.history[field].meta


@pytest.mark.parametrize(
    "method, inversion_method, freq",
    [
        ("bagg", "inverse_bagg", "15Min"),
        ("fagg", "inverse_fagg", "15Min"),
        ("nagg", "inverse_nagg", "15min"),
    ],
)
def test_resampleAutoInvert(data, method, inversion_method, freq):
    flags = initFlagsLike(data)
    field = data.columns[0]
    field_aligned = f"{field}_aligned"

    qc = SaQC(data, flags)
    qc = qc.resample(field=field, target=field_aligned, method=method, freq=freq)
    qc = qc.flagRange(field=field_aligned, min=0, max=100)
    qc_expected = qc.concatFlags(
        field=field_aligned, target=field, method=inversion_method
    )
    qc_got = qc.concatFlags(field=field_aligned, target=field, method="auto")

    _assertEqual(qc_got, qc_expected)
