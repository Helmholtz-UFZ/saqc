#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from pandas.core.indexing import IndexingError

from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags


def get_dataset(rows, cols):
    index = pd.date_range(start='2011-01-01', end='2011-01-10', periods=rows)
    df = pd.DataFrame(index=index)
    for c in range(cols):
        df[f'var{c}'] = np.linspace(0 + 100 * c, index.size, index.size)
    return df


field = 'var0'

DATASETS = [
    # get_dataset(0, 1),
    # get_dataset(1, 1),
    get_dataset(100, 1),
    # get_dataset(1000, 1),
    # get_dataset(0, 4),
    # get_dataset(1, 4),
    get_dataset(100, 4),
    # get_dataset(1000, 4),
    # get_dataset(10000, 40),
]

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_initFlags(data, flagger):
    flagger.initFlags(data)
    flags = flagger._flags
    assert isinstance(flags, pd.DataFrame)
    assert len(flags.index) == len(data.index)
    assert len(flags.columns) >= len(data.columns)


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_getFlags(data, flagger):

    # df
    flagger.initFlags(data)
    flags0 = flagger.getFlags()
    assert isinstance(flags0, pd.DataFrame)
    assert flags0.shape == data.shape
    assert (flags0.columns == data.columns).all()
    for dt in flags0.dtypes:
        assert isinstance(dt, pd.CategoricalDtype)

    # series
    flagger.initFlags(data)
    flags1 = flagger.getFlags(field)
    assert isinstance(flags1, pd.Series)
    assert isinstance(flags1.dtype, pd.CategoricalDtype)
    assert flags1.shape[0] == data.shape[0]
    assert flags1.name in data.columns

    # all the same
    flagger.initFlags(data)
    f0 = flagger.getFlags()[field]
    f1 = flagger.getFlags(field)
    assert np.all(f0 == f1)


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_isFlagged(data, flagger):
    # todo: add testcase with comparator

    # df
    flagger.initFlags(data)
    flagged0 = flagger.isFlagged()
    assert isinstance(flagged0, pd.DataFrame)
    assert flagged0.shape == data.shape
    assert (flagged0.columns == data.columns).all()
    for dt in flagged0.dtypes:
        assert is_bool_dtype(dt)

    # series
    flagger.initFlags(data)
    flagged1 = flagger.isFlagged(field)
    assert isinstance(flagged1, pd.Series)
    assert is_bool_dtype(flagged1.dtype)
    assert flagged1.shape[0] == data.shape[0]
    assert flagged1.name in data.columns

    # both the same
    assert (flagged0[field] == flagged1).all()

    # flag cannot be series
    flagger.initFlags(data)
    flag = pd.Series(index=data.index, data=flagger.BAD).astype(flagger.categories)
    try:
        flagger.isFlagged(field, flag=flag)
    except TypeError:
        pass
    else:
        raise AssertionError('this should not work')


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_setFlags(data, flagger):
    # todo flag=series

    flagger.initFlags(data)
    origin = flagger.getFlags()
    sl = slice('2011-01-02', '2011-01-05')

    # selection, return nevertheless the whole frame
    flags0 = flagger.setFlags(field, flag=flagger.GOOD, loc=sl)
    assert flags0.shape == origin.shape
    assert (flags0.columns == origin.columns).all()

    # all
    flagger.initFlags(data)
    flags0 = flagger.setFlags(field, flag=flagger.GOOD)
    assert flags0.shape == origin.shape
    assert (flags0.columns == origin.columns).all()
    flagged0 = flagger.getFlags(field)
    assert (flagged0 == flagger.GOOD).all()
    # overflag works BAD > GOOD
    flagger.setFlags(field, flag=flagger.BAD)
    flagged1 = flagger.getFlags(field)
    assert (flagged1 == flagger.BAD).all()
    # overflag dont work GOOD < BAD
    flagger.setFlags(field, flag=flagger.GOOD)
    flagged2 = flagger.getFlags(field)
    assert (flagged2 == flagger.BAD).all()  # still BAD
    # overflag do work with force
    flagger.setFlags(field, flag=flagger.GOOD, force=True)
    flagged3 = flagger.getFlags(field)
    assert (flagged3 == flagger.GOOD).all()

    # no field throw err
    try:
        flagger.setFlags()
    except TypeError:
        pass
    else:
        raise AssertionError
    try:
        flagger.setFlags(field=None)
    except ValueError:
        pass
    else:
        raise AssertionError


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_clearFlags(data, flagger):
    flagger.initFlags(data)
    origin = flagger.getFlags()
    sl = slice('2011-01-02', '2011-01-05')

    flagger.initFlags(data)
    flagger.setFlags(field, flag=flagger.BAD)
    flagged = flagger.isFlagged(field)
    assert len(flagged[flagged]) == len(data)
    cleared = flagger.clearFlags(field)
    assert cleared.shape == origin.shape
    assert (cleared.columns == origin.columns).all()
    flagged = flagger.isFlagged(field)
    assert len(flagged[flagged]) == 0

    # with chunk
    flagger.initFlags(data)
    flagger.setFlags(field, flag=flagger.BAD)
    flagged = flagger.isFlagged(field)
    assert len(flagged[flagged]) == len(data)
    cleared = flagger.clearFlags(field, loc=sl)
    assert cleared.shape == origin.shape
    assert (cleared.columns == origin.columns).all()
    flagged = flagger.isFlagged(field)
    assert len(flagged[flagged]) > 0
    unflagged = flagger.isFlagged(field, flag=flagger.UNFLAGGED, comparator='==')
    assert len(flagged[flagged]) + len(unflagged[unflagged]) == len(data)

    # no field throw err
    flagger.initFlags(data)
    try:
        flagger.clearFlags()
    except TypeError:
        pass
    else:
        raise AssertionError
    try:
        flagger.clearFlags(field=None)
    except ValueError:
        pass
    else:
        raise AssertionError


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_returnCopy(data, flagger):
    flagger.initFlags(data)
    origin = flagger.getFlags()

    f = flagger.getFlags()
    assert f is not origin
    f = flagger.isFlagged()
    assert f is not origin
    f = flagger.setFlags(field)
    assert f is not origin
    f = flagger.clearFlags(field)
    assert f is not origin


LOC_ILOC_FUNCS = [
    'isFlagged',
    'getFlags'
]


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('flaggerfunc', LOC_ILOC_FUNCS)
def test_loc(data, flagger, flaggerfunc):
    sl = slice('2011-01-02', '2011-01-05')
    chunk = data.loc[sl, field]
    d = data.loc[sl]
    m = data.index.get_loc(d.index[0])
    M = data.index.get_loc(d.index[-1])
    mask = np.full(len(data), False)
    mask[m:M] = True

    flagger_func = getattr(flagger, flaggerfunc)

    # masked
    flagger.initFlags(data)
    mflags0 = flagger_func(field, loc=mask)
    flagger.initFlags(data)
    mflags1 = flagger_func().loc[mask, field]
    flagger.initFlags(data)
    mflags2 = flagger_func(field).loc[mask]
    flagger.initFlags(data)
    mflags3 = flagger_func(loc=mask)[field]
    assert (mflags0 == mflags1).all()
    assert (mflags0 == mflags2).all()
    assert (mflags0 == mflags3).all()

    # indexed
    flagger.initFlags(data)
    iflags0 = flagger_func(field, loc=chunk.index)
    flagger.initFlags(data)
    iflags1 = flagger_func().loc[chunk.index, field]
    flagger.initFlags(data)
    iflags2 = flagger_func(field).loc[chunk.index]
    flagger.initFlags(data)
    iflags3 = flagger_func(loc=chunk.index)[field]
    assert (iflags0 == iflags1).all()
    assert (iflags0 == iflags2).all()
    assert (iflags0 == iflags3).all()

    # sliced
    flagger.initFlags(data)
    sflags0 = flagger_func(field, loc=sl)
    flagger.initFlags(data)
    sflags1 = flagger_func().loc[sl, field]
    flagger.initFlags(data)
    sflags2 = flagger_func(field).loc[sl]
    flagger.initFlags(data)
    sflags3 = flagger_func(loc=sl)[field]
    assert (sflags0 == sflags1).all()
    assert (sflags0 == sflags2).all()
    assert (sflags0 == sflags3).all()

    assert (sflags0 == iflags0).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('flaggerfunc', LOC_ILOC_FUNCS)
def test_iloc(data, flagger, flaggerfunc):
    M = len(data.index) - 1
    if M < 3:
        return
    m = M // 3
    M = m * 2

    array = data.reset_index(drop=True).index.values[m:M]
    sl = slice(m, M)
    mask = np.full(len(data), False)
    mask[sl] = True

    flagger_func = getattr(flagger, flaggerfunc)

    # masked
    flagger.initFlags(data)
    mflags0 = flagger_func(field, iloc=mask)
    flagger.initFlags(data)
    mflags1 = flagger_func().iloc[mask, 0]
    flagger.initFlags(data)
    mflags2 = flagger_func(field).iloc[mask]
    flagger.initFlags(data)
    mflags3 = flagger_func(iloc=mask)[field]
    assert (mflags0 == mflags1).all()
    assert (mflags0 == mflags2).all()
    assert (mflags0 == mflags3).all()

    # indexed
    flagger.initFlags(data)
    iflags0 = flagger_func(field, iloc=array)
    flagger.initFlags(data)
    iflags1 = flagger_func().iloc[array, 0]
    flagger.initFlags(data)
    iflags2 = flagger_func(field).iloc[array]
    flagger.initFlags(data)
    iflags3 = flagger_func(iloc=array)[field]
    assert (iflags0 == iflags1).all()
    assert (iflags0 == iflags2).all()
    assert (iflags0 == iflags3).all()

    # sliced
    flagger.initFlags(data)
    sflags0 = flagger_func(field, iloc=sl)
    flagger.initFlags(data)
    sflags1 = flagger_func().iloc[sl, 0]
    flagger.initFlags(data)
    sflags2 = flagger_func(field).iloc[sl]
    flagger.initFlags(data)
    sflags3 = flagger_func(iloc=sl)[field]
    assert (sflags0 == sflags1).all()
    assert (sflags0 == sflags2).all()
    assert (sflags0 == sflags3).all()

    assert (sflags0 == iflags0).all()
    assert (sflags0 == mflags0).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_classicUseCases(data, flagger):

    # data-mask, same length than flags
    flagger.initFlags(data)
    d = data[field]
    mask = d < (d.max() - d.min()) // 2
    flagger.setFlags(field, loc=mask, flag=flagger.BAD)
    flagged = flagger.isFlagged(field)
    assert (flagged == mask).all()

    # some fun with numpy but not same dimensions.. pass indices to iloc
    flagger.initFlags(data)
    indices = np.arange(0, len(data))
    mask = indices % 3 == 0
    indices = indices[mask]
    flagger.setFlags(field, iloc=indices, flag=flagger.BAD)
    flagged = flagger.isFlagged(field)
    assert (flagged.iloc[indices] == flagged[flagged]).all()
    unflagged = ~flagged


if __name__ == '__main__':
    flagger = TESTFLAGGERS[1]
    test_getFlags(flagger)
    print('done')
