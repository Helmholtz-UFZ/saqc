#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from test.common import TESTFLAGGER, initData
from saqc.flagger.backtrack import Backtrack

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


def test_append():
    pass


def test_squeeze():
    pass
