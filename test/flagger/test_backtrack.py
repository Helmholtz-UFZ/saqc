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
    this can be called for **any** BT and
    should never fail.
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


@pytest.mark.parametrize('data', data + [None])
def test_init(data: np.array):

    df = pd.DataFrame(data, dtype=float)
    bt = Backtrack(bt=df)

    check_invariants(bt)

    # shape would fail
    if data is not None:
        assert len(bt.index) == data.shape[0]
        assert len(bt.columns) == data.shape[1]
        assert bt.mask.all(axis=None)

    # check fastpath
    bt = Backtrack(bt=bt)
    check_invariants(bt)


@pytest.mark.parametrize('data', data + [None])
def test_init_with_mask(data: np.array):

    df = pd.DataFrame(data, dtype=float)

    bt = Backtrack(bt=df)

    check_invariants(bt)

    if data is None:
        return

    assert len(bt.index) == data.shape[0]
    assert len(bt.columns) == data.shape[1]
    assert bt.mask.all(axis=None)

