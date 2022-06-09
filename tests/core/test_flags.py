#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Dict, Union

import numpy as np
import pandas as pd
import pytest

import dios
from saqc.constants import UNFLAGGED
from saqc.core.flags import Flags
from tests.core.test_history import History
from tests.core.test_history import is_equal as hist_equal

_arrays = [
    np.array([[]]),
    np.zeros((1, 1)),
    np.zeros((3, 4)),
    np.ones((3, 4)),
    np.ones((3, 4)) * np.nan,
    np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    ),
    np.array(
        [
            [0, 0, 0, 0],
            [0, 1, np.nan, 3],
            [0, 1, 2, 3],
        ]
    ),
]

testdata = []
for d in _arrays:
    columns = list("abcdefgh")[: d.shape[1]]
    df = pd.DataFrame(d, dtype=float, columns=columns)
    dis = dios.DictOfSeries(df)
    di = {}
    di.update(df.items())
    testdata.append(df)
    testdata.append(di)
    testdata.append(dis)


def is_equal(f1, f2):
    assert f1.columns.equals(f2.columns)
    for c in f1.columns:
        assert hist_equal(f1.history[c], f2.history[c])


@pytest.mark.parametrize("data", testdata)
@pytest.mark.parametrize("copy", [True, False])
def test_init(data, copy):
    flags = Flags(data, copy=copy)
    assert isinstance(flags, Flags)
    assert len(data.keys()) == len(flags)


@pytest.mark.parametrize("data", testdata)
@pytest.mark.parametrize("copy", [True, False])
def test_init_from_other(data, copy):
    first = Flags(data)
    second = Flags(first, copy=copy)
    assert isinstance(second, Flags)
    assert first is not second
    if copy:
        assert first._data is not second._data
    else:
        assert first._data is second._data


@pytest.mark.parametrize(
    "data,msg",
    [
        ({1: pd.Series([1.0])}, "column names must be string"),
        ({"a": pd.Series([1, 2], dtype=int)}, "dtype must be float"),
    ],
)
def test_init_raise_ValueError(data, msg):
    with pytest.raises(ValueError) as e:
        Flags(data)
    assert msg in str(e.value)


@pytest.mark.parametrize(
    "data,msg",
    [
        ({"a": [1, 2]}, "cannot init from 'dict' of 'list'"),
    ],
)
def test_init_raise_TypeError(data, msg):
    with pytest.raises(TypeError) as e:
        Flags(data)
    assert msg in str(e.value)


@pytest.mark.parametrize("data", testdata)
def test_copy(data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]):
    flags = Flags(data)
    shallow = flags.copy(deep=False)
    deep = flags.copy(deep=True)

    # checks

    for copy in [deep, shallow]:
        assert isinstance(copy, Flags)
        assert copy is not flags
        assert copy._data is not flags._data
        for c in copy.columns:
            assert copy._data[c] is not flags._data[c]
        is_equal(copy, flags)

    assert deep is not shallow
    is_equal(deep, shallow)

    # the underling series data is the same
    for c in shallow.columns:
        assert shallow._data[c].index is flags._data[c].index

    # the underling series data was copied
    for c in deep.columns:
        assert deep._data[c].index is not flags._data[c].index


@pytest.mark.parametrize("data", testdata)
def test_flags_history(
    data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]
):
    flags = Flags(data)

    # get
    for c in flags.columns:
        hist = flags.history[c]
        assert isinstance(hist, History)
        assert len(hist) > 0

    # set
    for c in flags.columns:
        hist = flags.history[c]
        hlen = len(hist)
        hist.append(pd.Series(888.0, index=hist.index, dtype=float))
        flags.history[c] = hist
        assert isinstance(hist, History)
        assert len(hist) == hlen + 1


@pytest.mark.parametrize("data", testdata)
def test_get_flags(data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]):
    flags = Flags(data)

    for c in flags.columns:
        # check obvious
        var = flags[c]
        assert isinstance(var, pd.Series)
        assert not var.empty
        assert var.equals(flags._data[c].squeeze())

        # always a copy
        assert var is not flags[c]

        # in particular, a deep copy
        var[:] = 9999.0
        assert all(flags[c] != var)


@pytest.mark.parametrize("data", testdata)
def test_set_flags(data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]):
    flags = Flags(data)

    for c in flags.columns:
        var = flags[c]
        hlen = len(flags.history[c])
        new = pd.Series(9999.0, index=var.index, dtype=float)

        flags[c] = new
        assert len(flags.history[c]) == hlen + 1
        assert all(flags.history[c].squeeze() == 9999.0)
        assert all(flags.history[c].squeeze() == flags[c])

        # check if deep-copied correctly
        new[:] = 8888.0
        assert all(flags.history[c].squeeze() == 9999.0)

        # flags always overwrite former
        flags[c] = new
        assert len(flags.history[c]) == hlen + 2
        assert all(flags.history[c].squeeze() == 8888.0)
        assert all(flags.history[c].squeeze() == flags[c])

        # check if deep-copied correctly
        new[:] = 7777.0
        assert all(flags.history[c].squeeze() == 8888.0)


@pytest.mark.parametrize("data", testdata)
def test_set_flags_with_mask(
    data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]
):
    flags = Flags(data)

    for c in flags.columns:
        var = flags[c]
        mask = var == UNFLAGGED

        scalar = 222.0
        flags[mask, c] = scalar
        assert all(flags[c].loc[mask] == 222.0)
        assert all(flags[c].loc[~mask] != 222.0)

        # scalar without mask is not allowed, because
        # it holds to much potential to set the whole
        # column unintentionally.
        with pytest.raises(ValueError):
            flags[c] = 888.0

        vector = var.copy()
        vector[:] = 333.0
        flags[mask, c] = vector
        assert all(flags[c].loc[mask] == 333.0)
        assert all(flags[c].loc[~mask] != 333.0)

        # works with any that pandas eat, eg with numpy
        vector[:] = 444.0
        vector = vector.to_numpy()
        flags[mask, c] = vector
        assert all(flags[c].loc[mask] == 444.0)
        assert all(flags[c].loc[~mask] != 444.0)

        # test length miss-match (mask)
        if len(mask) > 1:
            wrong_len = mask[:-1]
            with pytest.raises(ValueError):
                flags[wrong_len, c] = vector

        # test length miss-match (value)
        if len(vector) > 1:
            wrong_len = vector[:-1]
            with pytest.raises(ValueError):
                flags[mask, c] = wrong_len


@pytest.mark.parametrize("data", testdata)
def test_set_flags_with_index(
    data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]
):
    flags = Flags(data)

    for c in flags.columns:
        var = flags[c]
        mask = var == UNFLAGGED
        index = mask[mask].index

        scalar = 222.0
        flags[index, c] = scalar
        assert all(flags[c].loc[mask] == 222.0)
        assert all(flags[c].loc[~mask] != 222.0)

        vector = var.copy()
        vector[:] = 333.0
        flags[index, c] = vector
        assert all(flags[c].loc[mask] == 333.0)
        assert all(flags[c].loc[~mask] != 333.0)

        # works with any that pandas eat, eg with numpy
        vector[:] = 444.0
        vector = vector.to_numpy()
        flags[index, c] = vector
        assert all(flags[c].loc[mask] == 444.0)
        assert all(flags[c].loc[~mask] != 444.0)

        # test length miss-match (value)
        if len(vector) > 1:
            wrong_len = vector[:-1]
            with pytest.raises(ValueError):
                flags[index, c] = wrong_len


def _validate_flags_equals_frame(flags, df):
    assert df.columns.equals(flags.columns)

    for c in flags.columns:
        assert df[c].index.equals(flags[c].index)
        assert df[c].equals(flags[c])  # respects nan's


@pytest.mark.parametrize("data", testdata)
def test_to_dios(data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]):
    flags = Flags(data)
    df = flags.toDios()

    assert isinstance(df, dios.DictOfSeries)
    _validate_flags_equals_frame(flags, df)


@pytest.mark.parametrize("data", testdata)
def test_to_frame(data: Union[pd.DataFrame, dios.DictOfSeries, Dict[str, pd.Series]]):
    flags = Flags(data)
    df = flags.toFrame()

    assert isinstance(df, pd.DataFrame)
    _validate_flags_equals_frame(flags, df)


@pytest.mark.parametrize(
    "columns",
    [["x", "y"], pd.Index(["x", "y"])],
)
def test_columns_setter(columns):
    flags = Flags(
        {"a": pd.Series([1, 2], dtype=float), "b": pd.Series([1, 2], dtype=float)}
    )
    flags.columns = columns
    for c in columns:
        assert c in flags.columns


@pytest.mark.parametrize(
    "columns,err",
    [
        ("foooo", TypeError),  # cannot cast to Index
        (pd.Index(["x", "x"]), TypeError),  # duplicates
        (pd.Index([1, 2]), TypeError),  # not string
        (pd.Index(["x", "y", "z"]), ValueError),  # wrong length
        (pd.Index(["x"]), ValueError),  # wrong length
    ],
)
def test_columns_setter_raises(columns, err):
    flags = Flags(
        {"a": pd.Series([1, 2], dtype=float), "b": pd.Series([1, 2], dtype=float)}
    )
    with pytest.raises(err):
        flags.columns = columns
