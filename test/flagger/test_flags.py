#!/usr/bin/env python
import dios
import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from test.common import TESTFLAGGER, initData
from saqc.flagger.flags import Flags

_data = [

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

data = []
for d in _data:
    columns = list('abcdefgh')[:d.shape[1]]
    df = pd.DataFrame(d, dtype=float, columns=columns)
    dis = dios.DictOfSeries(df)
    di = {}
    di.update(df.items())
    data.append(df)
    data.append(di)
    data.append(dis)


@pytest.mark.parametrize('data', data)
def test_init(data: np.array):
    flags = Flags(data)
    assert isinstance(flags, Flags)
    assert len(data.keys()) == len(flags)


def test_cache():
    arr = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    data = pd.DataFrame(arr, dtype=float, columns=list('abcd'))
    flags = Flags(data)

    # cache empty
    assert flags._cache == {}

    # invoke caching
    flags['a']
    assert 'a' in flags._cache

    # clears cache
    flags['a'] = pd.Series([0, 0, 0], dtype=float)
    assert 'a' not in flags._cache

    # cache all
    flags.to_dios()
    for c in flags.columns:
        assert c in flags._cache

    # cache survive renaming
    flags.columns = list('xyzq')
    for c in flags.columns:
        assert c in flags._cache

