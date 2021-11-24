#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd

from saqc.core.history import History, createHistoryFromData
from tests.common import dummyHistory

# see #GH143 combined backtrack
# (adjusted to current implementation)
example1 = (
    # flags
    np.array(
        [
            [0, np.nan, 50, 99, np.nan],
            [0, np.nan, 50, np.nan, 25],
            [0, 99, 99, 99, 25],
            [0, 99, np.nan, np.nan, 25],
        ]
    ),
    # expected from max()
    np.array([99, 25, 25, 25]),
)

# see #GH143
example2 = (
    # flags
    np.array(
        [
            [0, 99, np.nan, 0],
            [0, np.nan, 99, np.nan],
            [0, np.nan, np.nan, np.nan],
            [0, np.nan, np.nan, 0],
        ]
    ),
    # expected from max()
    np.array([0, 99, 0, 0]),
)

data = [
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


def check_invariants(hist):
    """
    This can be called for **any** FH.
    The assertions must hold in any case.
    """
    # basics
    assert isinstance(hist, History)
    assert isinstance(hist.hist, pd.DataFrame)
    assert isinstance(hist.meta, list)
    assert all(
        [isinstance(dtype, (float, pd.CategoricalDtype)) for dtype in hist.hist.dtypes]
    )
    assert all([isinstance(e, dict) for e in hist.meta])
    assert hist.columns is hist.hist.columns
    assert hist.index is hist.hist.index
    assert len(hist) == len(hist.columns) == len(hist.meta)

    # advanced
    assert hist.columns.equals(pd.Index(range(len(hist))))
    assert isinstance(hist.max(), pd.Series)


def is_equal(hist1: History, hist2: History):
    """
    Check if two FH are (considered) equal, namely have equal 'hist'
    """
    return hist1.hist.equals(hist2.hist)


@pytest.mark.parametrize("data", data + [None])
def test_init(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    hist = History(df.index)
    check_invariants(hist)


@pytest.mark.parametrize("data", data + [None])
def test_createHistory(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    meta = [{}] * len(df.columns)
    hist = createHistoryFromData(df, meta)

    check_invariants(hist)

    # shape would fail
    if data is not None:
        assert len(hist.index) == data.shape[0]
        assert len(hist.columns) == data.shape[1]


@pytest.mark.parametrize("data", data + [None])
def test_copy(data):
    # init
    df = pd.DataFrame(data, dtype=float)
    hist = History(df.index)
    for _, s in df.items():
        hist.append(s)
    shallow = hist.copy(deep=False)
    deep = hist.copy(deep=True)

    # checks
    for copy in [deep, shallow]:
        check_invariants(copy)
        assert copy is not hist
        assert copy.hist is not hist.hist
        assert copy.meta is not hist.meta
        assert is_equal(copy, hist)

    assert deep is not shallow
    assert is_equal(deep, shallow)

    # underling pandas data was only copied with deep=True
    assert shallow.hist.index is hist.hist.index
    assert deep.hist.index is not hist.hist.index


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("data", data + [None])
def test_reindex_trivial_cases(data, copy):
    df = pd.DataFrame(data, dtype=float)
    orig = dummyHistory(hist=df)

    # checks
    for index in [df.index, pd.Index([])]:
        ref = orig.copy()
        hist = ref.reindex(index, copy=copy)
        if copy:
            assert hist is not ref
        else:
            assert hist is ref
        check_invariants(hist)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("data", data + [None])
def test_reindex_missing_indicees(data, copy):
    df = pd.DataFrame(data, dtype=float)
    orig = dummyHistory(hist=df)
    index = df.index[1:-1]
    hist = orig.reindex(index, copy=copy)
    if copy:
        assert hist is not orig
    else:
        assert hist is orig
    check_invariants(hist)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("data", data + [None])
def test_reindex_extra_indicees(data, copy):
    df = pd.DataFrame(data, dtype=float)
    orig = dummyHistory(hist=df)
    index = df.index.append(pd.Index(range(len(df.index), len(df.index) + 5)))
    hist = orig.reindex(index, copy=copy)
    if copy:
        assert hist is not orig
    else:
        assert hist is orig
    check_invariants(hist)


@pytest.mark.parametrize(
    "s, meta",
    [
        (pd.Series(0, index=range(6), dtype=float), None),
        (pd.Series(0, index=range(6), dtype=float), {}),
        (pd.Series(1, index=range(6), dtype=float), {"foo": "bar"}),
    ],
)
def test_append_with_meta(s, meta):
    hist = History(s.index)
    hist.append(s, meta=meta)
    check_invariants(hist)

    if meta is None:
        meta = {}

    assert hist.meta[0] is not meta
    assert hist.meta == [meta]

    hist.append(s, meta=meta)
    check_invariants(hist)
    assert hist.meta == [meta, meta]


@pytest.fixture(scope="module")
def __hist():
    # this FH is filled by
    #  - test_append
    #  - test_append_force
    return History(index=pd.Index(range(6)))


# this test append more rows to the resulting
# FH from the former test
@pytest.mark.parametrize(
    "s, max_val",
    [
        (pd.Series(0, index=range(6), dtype=float), 0),
        (pd.Series(1, index=range(6), dtype=float), 1),
        (pd.Series(np.nan, index=range(6), dtype=float), 1),
        (pd.Series(0, index=range(6), dtype=float), 0),
    ],
)
def test_append_force(__hist, s, max_val):
    hist = __hist
    hist.append(s)
    check_invariants(hist)
    assert all(hist.max() == max_val)
