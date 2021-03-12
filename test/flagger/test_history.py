#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.history import History

# see #GH143 combined backtrack
# (adjusted to current implementation)
example1 = (

    # flags
    np.array([
        [0, np.nan, 50, 99, np.nan],
        [0, np.nan, 50, np.nan, 25],
        [0, 99, 99, 99, 25],
        [0, 99, np.nan, np.nan, 25],
    ]),

    # mask
    np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]),

    # expected from max()
    np.array([99, 25, 25, 25])
)

# see #GH143
example2 = (

    # flags
    np.array([
        [0, 99, np.nan, 0],
        [0, np.nan, 99, np.nan],
        [0, np.nan, np.nan, np.nan],
        [0, np.nan, np.nan, 0],
    ]),

    # mask
    np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1],
    ]),

    # expected from max()
    np.array([0, 99, 0, 0])
)

data = [

    np.array([[]]),
    np.zeros((1, 1)),
    np.zeros((3, 4)),
    np.ones((3, 4)),
    np.ones((3, 4)) * np.nan,

    np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]),

    np.array([
        [0, 0, 0, 0],
        [0, 1, np.nan, 3],
        [0, 1, 2, 3],
    ]),
]


def check_invariants(hist):
    """
    This can be called for **any** FH.
    The assertions must hold in any case.
    """
    # basics
    assert isinstance(hist, History)
    assert isinstance(hist.hist, pd.DataFrame)
    assert isinstance(hist.mask, pd.DataFrame)
    assert all(hist.hist.dtypes == float)
    assert all(hist.mask.dtypes == bool)
    assert hist.hist.columns.equals(hist.mask.columns)
    assert hist.columns is hist.hist.columns
    assert hist.index is hist.hist.index
    assert len(hist) == len(hist.columns)

    # advanced
    assert hist.columns.equals(pd.Index(range(len(hist))))
    assert isinstance(hist.max(), pd.Series)
    assert hist.mask.empty or hist.mask.iloc[:, -1].all()

    # False propagation

    # for each row this must hold:
    # either the row has one change (False->True)
    # or the entire row is True
    if not hist.empty:
        idxmax = hist.mask.idxmax(axis=1)
        print(f'idxmax: {idxmax}')
        for row, col in idxmax.items():
            # this is contra intuitive, it gets the positional (for iloc)
            row = idxmax.index.get_loc(row)

            assert all(hist.mask.iloc[row, :col] == False)
            assert all(hist.mask.iloc[row, col:] == True)


def is_equal(hist1: History, hist2: History):
    """
    Check if two FH are (considered) equal, namely
    have equal 'hist' and equal 'mask'.
    """
    return hist1.hist.equals(hist2.hist) and hist1.mask.equals(hist2.mask)


@pytest.mark.parametrize('data', data + [None])
def test_init(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    hist = History(hist=df)

    check_invariants(hist)

    # shape would fail
    if data is not None:
        assert len(hist.index) == data.shape[0]
        assert len(hist.columns) == data.shape[1]
        assert hist.mask.all(axis=None)

    # check fastpath
    fast = History(hist=hist)
    check_invariants(fast)

    assert is_equal(hist, fast)


@pytest.mark.parametrize('data', data + [None])
def test_init_with_mask(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    mask = pd.DataFrame(data, dtype=bool)
    if not mask.empty:
        mask.iloc[:, -1] = True
    hist = History(hist=df, mask=mask)

    check_invariants(hist)

    # shape would fail
    if data is not None:
        assert len(hist.index) == data.shape[0]
        assert len(hist.columns) == data.shape[1]

    # check fastpath
    fast = History(hist=hist)
    check_invariants(fast)

    assert is_equal(hist, fast)


@pytest.mark.parametrize('data', data + [None])
def test_copy(data):
    # init
    df = pd.DataFrame(data, dtype=float)
    hist = History(hist=df)
    shallow = hist.copy(deep=False)
    deep = hist.copy(deep=True)

    # checks

    for copy in [deep, shallow]:
        check_invariants(copy)
        assert copy is not hist
        assert is_equal(copy, hist)

    assert deep is not shallow
    assert is_equal(deep, shallow)

    assert deep.hist is not hist.hist
    assert deep.mask is not hist.mask
    assert shallow.hist is hist.hist
    assert shallow.mask is hist.mask


@pytest.mark.parametrize('data', data + [None])
def test_reindex_trivial_cases(data):
    df = pd.DataFrame(data, dtype=float)
    orig = History(hist=df)

    # checks
    for index in [df.index, pd.Index([])]:
        hist = orig.copy()
        ref = hist.reindex(index)
        assert ref is hist  # check if working inplace
        check_invariants(hist)


@pytest.mark.parametrize('data', data + [None])
def test_reindex_missing_indicees(data):
    df = pd.DataFrame(data, dtype=float)
    hist = History(hist=df)
    index = df.index[1:-1]
    # checks
    ref = hist.reindex(index)
    assert ref is hist  # check if working inplace
    check_invariants(hist)


@pytest.mark.parametrize('data', data + [None])
def test_reindex_extra_indicees(data):
    df = pd.DataFrame(data, dtype=float)
    hist = History(hist=df)
    index = df.index.append(pd.Index(range(len(df.index), len(df.index) + 5)))
    # checks
    ref = hist.reindex(index)
    assert ref is hist  # check if working inplace
    check_invariants(hist)


@pytest.fixture(scope='module')
def __hist():
    # this FH is filled by
    #  - test_append
    #  - test_append_force
    return History()


@pytest.mark.parametrize('s, max_val', [
    (pd.Series(0, index=range(6), dtype=float), 0),
    (pd.Series(1, index=range(6), dtype=float), 1),
    (pd.Series(np.nan, index=range(6), dtype=float), 1),
    (pd.Series(1, index=range(6), dtype=float), 1),
    (pd.Series(0, index=range(6), dtype=float), 1),
])
def test_append(__hist, s, max_val):
    hist = __hist
    hist.append(s, force=False)
    check_invariants(hist)
    assert all(hist.max() == max_val)


# this test append more rows to the resulting
# FH from the former test
@pytest.mark.parametrize('s, max_val', [
    (pd.Series(0, index=range(6), dtype=float), 0),
    (pd.Series(1, index=range(6), dtype=float), 1),
    (pd.Series(np.nan, index=range(6), dtype=float), 1),
    (pd.Series(0, index=range(6), dtype=float), 0),
])
def test_append_force(__hist, s, max_val):
    hist = __hist
    hist.append(s, force=True)
    check_invariants(hist)
    assert all(hist.max() == max_val)


def test_squeeze():
    # init
    d, m, exp = example2
    d = pd.DataFrame(d, dtype=float)
    m = pd.DataFrame(m, dtype=bool)
    orig = History(hist=d, mask=m)

    check_invariants(orig)
    assert all(orig.max() == exp)

    # checks

    for n in range(len(orig) + 1):
        hist = orig.copy()
        hist.squeeze(n)

        check_invariants(hist)

        # squeeze for less then 2 rows does nothing
        if n < 2:
            assert is_equal(hist, orig)
        else:
            assert len(hist) == len(orig) - n + 1

        # result does not change
        assert all(hist.max() == exp)
        print(hist)
