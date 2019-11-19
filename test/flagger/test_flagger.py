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
    # BaseFlagger(['NIL', 'GOOD', 'BAD']),
    # DmpFlagger(),
    SimpleFlagger()
]


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_initFlags(data, flagger):
    flags = flagger.initFlags(data).getFlags()
    assert isinstance(flags, pd.DataFrame)
    assert len(flags.index) == len(data.index)
    assert len(flags.columns) >= len(data.columns)


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_getFlags(data, flagger):
    flagger.initFlags(data)

    # df
    flags0 = flagger.getFlags()
    assert isinstance(flags0, pd.DataFrame)
    assert flags0.shape == data.shape
    assert (flags0.columns == data.columns).all()
    for dt in flags0.dtypes:
        assert isinstance(dt, pd.CategoricalDtype)

    # series
    flags1 = flagger.getFlags(field)
    assert isinstance(flags1, pd.Series)
    assert isinstance(flags1.dtype, pd.CategoricalDtype)
    assert flags1.shape[0] == data.shape[0]
    assert flags1.name in data.columns

    # all the same
    # NOTE: doesn't make sense here
    # flags2 = flagger.getFlags(flags[[field]]).squeeze()
    # assert (flags0[field] == flags1).all()
    # assert (flags0[field] == flags2).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_isFlagged(data, flagger):
    # todo: add testcase with comparator

    flagger.initFlags(data)

    # df
    flagged0 = flagger.isFlagged()
    assert isinstance(flagged0, pd.DataFrame)
    assert flagged0.shape == data.shape
    assert (flagged0.columns == data.columns).all()
    for dt in flagged0.dtypes:
        assert is_bool_dtype(dt)

    # series
    flagged1 = flagger.isFlagged(field)
    assert isinstance(flagged1, pd.Series)
    assert is_bool_dtype(flagged1.dtype)
    assert flagged1.shape[0] == data.shape[0]
    assert flagged1.name in data.columns

    # both the same
    assert (flagged0[field] == flagged1).all()

    # # flag cannot be series
    # # NOTE: doesn't make sense here
    # flag = pd.Series(index=data.index, data=flagger.BAD).astype(flagger.categories)
    # try:
    #     flagger.isFlagged(flags, field=field, flag=flag)
    # except TypeError:
    #     pass
    # else:
    #     raise AssertionError('this should not work')


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_setFlags(data, flagger):
    base = flagger.initFlags(data).getFlags()
    sl = slice('2011-01-02', '2011-01-05')

    flags0 = flagger.setFlags(field, flag=flagger.GOOD, loc=sl).getFlags()
    assert flags0.shape == base.shape
    assert (flags0.columns == base.columns).all()
    assert (flags0.loc[sl, field] == flagger.GOOD).all()

    # overflag works BAD > GOOD
    flags1 = flagger.setFlags(field, flag=flagger.BAD).getFlags(field)
    assert (flags1 == flagger.BAD).all()

    # overflag doesn't work GOOD < BAD
    flags2 = flagger.setFlags(field, flag=flagger.GOOD).getFlags(field)
    assert (flags2 == flagger.BAD).all()  # still BAD

    # overflag does work with force
    flags3 = flagger.setFlags(field, flag=flagger.GOOD, force=True).getFlags(field)
    assert (flags3 == flagger.GOOD).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_clearFlags(data, flagger):

    flagger.initFlags(data)
    origin = flagger.getFlags()
    sl = slice('2011-01-02', '2011-01-05')

    flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(origin)

    flagger.clearFlags(field)
    assert np.sum(flagger.isFlagged(field)) == 0

    flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(origin)

    flagger.clearFlags(field, loc=sl)
    unflagged = flagger.isFlagged(field, loc=sl)
    assert np.sum(unflagged) == 0
    assert np.sum(flagger.isFlagged(field)) == len(data) - len(unflagged)


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_returnCopy(data, flagger):
    flagger.initFlags(data)
    origin = flagger.getFlags()

    f = flagger.getFlags()
    assert f is not origin
    f = flagger.isFlagged()
    assert f is not origin
    f = flagger.setFlags(field).getFlags()
    assert f is not origin
    f = flagger.clearFlags(field).getFlags()
    assert f is not origin


LOC_ILOC_FUNCS = [
    'isFlagged',
    'getFlags'
]


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('flaggerfunc', LOC_ILOC_FUNCS)
def test_loc(data, flagger, flaggerfunc):
    flagger.initFlags(data)
    flags = flagger.getFlags()
    sl = slice('2011-01-02', '2011-01-05')
    chunk = data.loc[sl, field]
    d = data.loc[sl]
    m = data.index.get_loc(d.index[0])
    M = data.index.get_loc(d.index[-1])
    mask = np.full(len(data), False)
    mask[m:M] = True

    flagger_func = getattr(flagger, flaggerfunc)

    # masked
    mflags0 = flagger_func(field, loc=mask)
    mflags1 = flagger_func().loc[mask, field]
    mflags2 = flagger_func(field).loc[mask]
    mflags3 = flagger_func(loc=mask)[field]
    assert (mflags0 == mflags1).all()
    assert (mflags0 == mflags2).all()
    assert (mflags0 == mflags3).all()

    # indexed
    iflags0 = flagger_func(field, loc=chunk.index)
    iflags1 = flagger_func().loc[chunk.index, field]
    iflags2 = flagger_func(field).loc[chunk.index]
    iflags3 = flagger_func(loc=chunk.index)[field]
    assert (iflags0 == iflags1).all()
    assert (iflags0 == iflags2).all()
    assert (iflags0 == iflags3).all()

    # sliced
    sflags0 = flagger_func(field, loc=sl)
    sflags1 = flagger_func().loc[sl, field]
    sflags2 = flagger_func(field).loc[sl]
    sflags3 = flagger_func(loc=sl)[field]
    assert (sflags0 == sflags1).all()
    assert (sflags0 == sflags2).all()
    assert (sflags0 == sflags3).all()

    assert (sflags0 == iflags0).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('flaggerfunc', LOC_ILOC_FUNCS)
def test_iloc(data, flagger, flaggerfunc):
    flagger.initFlags(data)
    flags = flagger.getFlags()
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
    mflags0 = flagger_func(field, iloc=mask)
    mflags1 = flagger_func().iloc[mask, 0]
    mflags2 = flagger_func(field).iloc[mask]
    mflags3 = flagger_func(iloc=mask)[field]
    assert (mflags0 == mflags1).all()
    assert (mflags0 == mflags2).all()
    assert (mflags0 == mflags3).all()

    # indexed
    iflags0 = flagger_func(field, iloc=array)
    iflags1 = flagger_func().iloc[array, 0]
    iflags2 = flagger_func(field).iloc[array]
    iflags3 = flagger_func(iloc=array)[field]
    assert (iflags0 == iflags1).all()
    assert (iflags0 == iflags2).all()
    assert (iflags0 == iflags3).all()

    # sliced
    sflags0 = flagger_func(field, iloc=sl)
    sflags1 = flagger_func().iloc[sl, 0]
    sflags2 = flagger_func(field).iloc[sl]
    sflags3 = flagger_func(iloc=sl)[field]
    assert (sflags0 == sflags1).all()
    assert (sflags0 == sflags2).all()
    assert (sflags0 == sflags3).all()

    assert (sflags0 == iflags0).all()
    assert (sflags0 == mflags0).all()


@pytest.mark.parametrize('data', DATASETS)
@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_classicUseCases(data, flagger):
    flagger.initFlags(data)
    flags = flagger.getFlags()

    # data-mask, same length than flags
    d = data[field]
    mask = d < (d.max() - d.min()) // 2
    flagger.clearFlags(field)
    flagged = flagger.setFlags(field, loc=mask, flag=flagger.BAD).isFlagged(field)
    assert (flagged == mask).all()

    # some fun with numpy but not same dimensions.. pass indices to iloc
    indices = np.arange(0, len(data))
    mask = indices % 3 == 0
    indices = indices[mask]
    flagger.clearFlags(field)
    flagged = flagger.setFlags(field, iloc=indices, flag=flagger.BAD).isFlagged(field)
    assert (flagged.iloc[indices] == flagged[flagged]).all()
