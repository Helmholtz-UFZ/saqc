#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from test.common import TESTFLAGGER, initData
from saqc.flagger.backtrack import Backtrack

# see #GH143 combined bt
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


def check_invariants(bt):
    """
    This can be called for **any** BT.
    The assertions must hold in any case.
    """
    # basics
    assert isinstance(bt, Backtrack)
    assert isinstance(bt.bt, pd.DataFrame)
    assert isinstance(bt.mask, pd.DataFrame)
    assert all(bt.bt.dtypes == float)
    assert all(bt.mask.dtypes == bool)
    assert bt.bt.columns.equals(bt.mask.columns)
    assert bt.columns is bt.bt.columns
    assert bt.index is bt.bt.index
    assert len(bt) == len(bt.columns)

    # advanced
    assert bt.columns.equals(pd.Index(range(len(bt))))
    assert isinstance(bt.max(), pd.Series)
    assert bt.mask.empty or bt.mask.iloc[:, -1].all()

    # False propagation

    # for each row this must hold:
    # either the row has one change (False->True)
    # or the entire row is True
    if not bt.empty:
        idxmax = bt.mask.idxmax(axis=1)
        for row, col in idxmax.items():
            assert all(bt.mask.iloc[row, :col] == False)
            assert all(bt.mask.iloc[row, col:] == True)


def is_equal(bt1: Backtrack, bt2: Backtrack):
    """
    Check if two BT are (considered) equal, namely
    have equal 'bt' and equal 'mask'.
    """
    return bt1.bt.equals(bt2.bt) and bt1.mask.equals(bt2.mask)


@pytest.mark.parametrize('data', data + [None])
def test_init(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    bt = Backtrack(bt=df)

    check_invariants(bt)

    # shape would fail
    if data is not None:
        assert len(bt.index) == data.shape[0]
        assert len(bt.columns) == data.shape[1]
        assert bt.mask.all(axis=None)

    # check fastpath
    fast = Backtrack(bt=bt)
    check_invariants(fast)

    assert is_equal(bt, fast)


@pytest.mark.parametrize('data', data + [None])
def test_init_with_mask(data: np.array):
    # init
    df = pd.DataFrame(data, dtype=float)
    mask = pd.DataFrame(data, dtype=bool)
    if not mask.empty:
        mask.iloc[:, -1] = True
    bt = Backtrack(bt=df, mask=mask)

    check_invariants(bt)

    # shape would fail
    if data is not None:
        assert len(bt.index) == data.shape[0]
        assert len(bt.columns) == data.shape[1]

    # check fastpath
    fast = Backtrack(bt=bt)
    check_invariants(fast)

    assert is_equal(bt, fast)


@pytest.mark.parametrize('data', data + [None])
def test_copy(data):
    # init
    df = pd.DataFrame(data, dtype=float)
    bt = Backtrack(bt=df)
    shallow = bt.copy(deep=False)
    deep = bt.copy(deep=True)

    # checks

    for copy in [deep, shallow]:
        check_invariants(copy)
        assert copy is not bt
        assert is_equal(copy, bt)

    assert deep is not shallow
    assert is_equal(deep, shallow)

    assert deep.bt is not bt.bt
    assert deep.mask is not bt.mask
    assert shallow.bt is bt.bt
    assert shallow.mask is bt.mask


@pytest.fixture(scope='module')
def __bt():
    # this BT is filled by
    #  - test_append
    #  - test_append_force
    return Backtrack()


@pytest.mark.parametrize('s, max_val', [
    (pd.Series(val, index=range(6), dtype=float), max_val)
    for val, max_val
    in zip(
        [0, 1, np.nan, 1, 0],
        [0, 1, 1, 1, 1]  # expected max-val
    )
])
def test_append(__bt, s, max_val):
    bt = __bt
    bt.append(s, force=False)
    check_invariants(bt)
    assert all(bt.max() == max_val)


# this test append more rows to the resulting
# BT from the former test
@pytest.mark.parametrize('s, max_val', [
    (pd.Series(val, index=range(6), dtype=float), max_val)
    for val, max_val
    in zip(
        [0, 1, np.nan, 0],
        [0, 1, 1, 0],  # expected max-val
    )
])
def test_append_force(__bt, s, max_val):
    bt = __bt
    bt.append(s, force=True)
    check_invariants(bt)
    assert all(bt.max() == max_val)


def test_squeeze():
    # init
    d, m, exp = example2
    d = pd.DataFrame(d, dtype=float)
    m = pd.DataFrame(m, dtype=bool)
    orig = Backtrack(bt=d, mask=m)

    check_invariants(orig)
    assert all(orig.max() == exp)

    # checks

    for n in range(len(orig)):
        bt = orig.copy()
        bt.squeeze(n)

        check_invariants(bt)

        # squeeze for less then 2 rows does nothing
        if n < 2:
            assert is_equal(bt, orig)
        else:
            assert len(bt) == len(orig) - n + 1

        # result does not change
        assert all(bt.max() == exp)
        print(bt)
