#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-


import uuid

import numpy as np
import pandas as pd
import pytest

from saqc import SaQC
from saqc.core.flags import initFlagsLike
from saqc.funcs.resampling import (
    _aggregationGrouper,
    _constructAggregationReindexer,
    _constructRollingReindexer,
    _reindexer,
    _rollingReindexer,
)
from saqc.lib.tools import toSequence
from tests.common import checkInvariants
from tests.fixtures import data


def _is_uuid(u):
    try:
        uuid.UUID(u.__str__())
        return True
    except ValueError:
        return False


def _uuidFieldFilter(meta):
    "replace seeded temporary fieldnames in meta for comparison"
    meta_out = meta.copy()
    field_dict = {}
    id_count = 0
    for m in range(len(meta)):
        field = meta_out[m].get("kwargs", {}).get("field", None)
        if field is not None:
            is_list = isinstance(field, list)
            field = toSequence(field)
            field_dict = dict(
                {
                    f: f"__UUIDFIELD{k}_{id_count}" if _is_uuid(f) else f
                    for k, f in enumerate(field)
                },
                **field_dict,
            )
            field = [field_dict[f] for f in field]
            meta_out[m]["kwargs"]["field"] = field if is_list else field[0]
            id_count += 1
    return meta_out


def _test_flagsSurviveReshaping():
    """
    flagging -> reshaping -> test (flags also was reshaped correctly)
    """
    pass


def _test_flagsSurviveInverseReshaping():
    """
    inverse reshaping -> flagging -> test (flags also was reshaped correctly)"""
    pass


def _test_flagsSurviveBackprojection():
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
                data=[np.nan, -87.5, -25.0, 0.0, 37.5, 50.0],
                index=pd.date_range(
                    "2010-12-31 23:45:00", "2011-01-01 01:00:00", freq="15min"
                ),
            ),
        ),
        (
            "nagg",
            "30Min",
            pd.Series(
                data=[np.nan, -87.5, -25.0, 87.5],
                index=pd.date_range(
                    "2010-12-31 23:30:00", "2011-01-01 01:00:00", freq="30min"
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
    qc = qc.resample(field_aggregated, freq, func="sum", method=method)
    assert qc._data[field_aggregated].index.freq == pd.Timedelta(freq)
    assert qc._data[field_aggregated].equals(expected)
    assert qc._flags.history[field_aggregated].meta[-1]["func"] == "resample"
    checkInvariants(qc._data, qc._flags, field_aggregated, identical=True)

    qc = qc.concatFlags(field_aggregated, target=field, method=method, invert=True)
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
    qc = qc.align(field_aligned, freq=freq, method=method)
    assert qc.data[field_aligned].equals(expected)
    checkInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.concatFlags(field_aligned, target=field, method="mshift", invert=True)
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
                data=[np.nan, -37.5, -25.0, 0.0, 37.5, 50.0],
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
                data=[np.nan, -37.5, 0.0, 50.0],
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
    assert (meta["func"], meta["kwargs"]["method"]) == ("reindex", method)
    checkInvariants(qc._data, qc._flags, field, identical=True)

    qc = qc.concatFlags(field_aligned, target=field, method=method, invert=True)
    assert qc.data[field].equals(pre_data[field])
    assert qc.flags[field].equals(pre_flags[field])
    checkInvariants(qc._data, qc._flags, field, identical=True)


@pytest.mark.parametrize(
    "method, freq",
    [
        ("linear", "15min"),
        ("bshift", "15Min"),
        ("fshift", "15Min"),
        ("nshift", "15min"),
        ("pad", "15min"),
    ],
)
def test_alignAutoInvert(data, method, freq):
    flags = initFlagsLike(data)
    field = data.columns[0]
    field_aligned = f"{field}_aligned"

    qc = SaQC(data, flags)
    qc = qc.align(field=field, target=field_aligned, method=method, freq=freq)
    qc = qc.flagDummy(field=field_aligned)
    qc_expected = qc.concatFlags(
        field=field_aligned, target=field, method=method, invert=True
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
            field=field_aligned,
            target=field,
            method="mshift",
            invert=True,
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
                field=field_aligned,
                method="mshift",
                invert=True,
            ).concatFlags(
                field=field_aligned, target=field, method="fshift", invert=True
            )
        ),
    )


def _assertEqual(left: SaQC, right: SaQC):
    for field in left.data.columns:
        assert left._data[field].equals(right._data[field])
        assert left._flags[field].equals(right._flags[field])
        assert left._flags.history[field].hist.equals(right._flags.history[field].hist)
        assert _uuidFieldFilter(left._flags.history[field].meta) == _uuidFieldFilter(
            right._flags.history[field].meta
        )


@pytest.mark.parametrize(
    "method, freq",
    [
        ("bagg", "15Min"),
        ("fagg", "15Min"),
        ("nagg", "15min"),
    ],
)
def test_resampleAutoInvert(data, method, freq):
    flags = initFlagsLike(data)
    field = data.columns[0]
    field_aligned = f"{field}_aligned"

    qc = SaQC(data, flags)
    qc = qc.resample(field=field, target=field_aligned, method=method, freq=freq)
    qc = qc.flagRange(field=field_aligned, min=0, max=100)
    qc_expected = qc.concatFlags(
        field=field_aligned, target=field, method=method, invert=True
    )
    qc_got = qc.concatFlags(field=field_aligned, target=field, method="auto")

    _assertEqual(qc_got, qc_expected)


@pytest.mark.parametrize(
    "method, expected",
    [("froll", [1, 1, 1, 2]), ("nroll", [2, 2, 3, 3]), ("broll", [3, 3, 3, 3])],
)
def test_constructRollingReindexer(method, expected):
    target_idx = pd.date_range("2000", periods=4, freq="4.5min")
    datser = pd.Series(
        np.ones(10), index=pd.date_range("2000", periods=10, freq="10min"), name="data"
    )
    reindexer = _constructRollingReindexer(
        "sum", target_idx, window="30min", direction=method
    )
    result = reindexer(datser)
    assert np.all(result.index == target_idx)
    assert result.to_list() == expected


@pytest.mark.parametrize(
    "center, fwd, expected",
    [(False, 1, [1, 1, 1, 2]), (True, 1, [2, 2, 3, 3]), (False, -1, [3, 3, 3, 3])],
)
def test_rollingReindexer(center, fwd, expected):
    target_idx = pd.date_range("2000", periods=4, freq="4.5min")
    datser = pd.Series(
        np.ones(10), index=pd.date_range("2000", periods=10, freq="10min"), name="data"
    )
    result = _rollingReindexer(datser, target_idx, "sum", "30min", center, fwd)
    assert np.all(result.index == target_idx)
    assert result.to_list() == expected


def test_constructAggregationReindexer():
    target_idx = pd.date_range("2000", periods=4, freq="4.5min")
    datser = pd.Series(
        np.ones(10), index=pd.date_range("2000", periods=10, freq="10min"), name="data"
    )
    grouper = [0, 0, 0, 0, 2, 2, 3, 3, 3, 3]
    reindexer = _constructAggregationReindexer("sum", target_idx, grouper, np.nan)
    result = reindexer(datser)
    assert result.index.equals(target_idx)
    assert np.array_equal(result.values, [4, np.nan, 2, 4], equal_nan=True)


def test_reindexer():
    target_idx = pd.date_range("2000", periods=4, freq="4.5min")
    datser = pd.Series(
        np.ones(10), index=pd.date_range("2000", periods=10, freq="10min"), name="data"
    )
    grouper = [0, 0, 0, 0, 2, 2, 3, 3, 3, 3]
    result = _reindexer(datser, "sum", target_idx, grouper, np.nan)
    assert result.index.equals(target_idx)
    assert np.array_equal(result.values, [4, np.nan, 2, 4], equal_nan=True)


@pytest.mark.parametrize(
    "method, tolerance, expected",
    [
        ("bshift", "4.5Min", [0, 2, 4, np.nan]),
        ("bshift", "20Min", [0, 2, 4, 4]),
        ("nshift", "4.5min", [0, 2, 4]),
        ("nshift", "1min", [0, 2]),
        ("fshift", "1min", [0, np.nan, np.nan, np.nan]),
        ("nagg", "1min", [0, 2, np.nan, np.nan]),
        ("bagg", "30min", [0, 2, 4, 4]),
        ("match", None, [0, np.nan, np.nan, np.nan]),
    ],
)
def test_aggregationGrouper(method, tolerance, expected):
    target_idx = pd.date_range("2000", periods=5, freq="4.5min")
    idx_source = pd.date_range("2000", periods=4, freq="10min")
    datser = pd.Series(np.ones(4), index=idx_source, name="data")
    result, _ = _aggregationGrouper(
        method, target_idx, idx_source, tolerance, datser, False
    )
    assert (method == "nshift") or (result.index.equals(idx_source))
    assert np.array_equal(result.values, expected, equal_nan=True)
