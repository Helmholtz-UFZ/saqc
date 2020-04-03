#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from test.common import TESTFLAGGER


def _getDataset(rows, cols):
    df = pd.DataFrame()
    for c in range(cols):
        df[f"var{c}"] = np.linspace(0 + 100 * c, rows, rows)
    vals = pd.date_range(start="2011-01-01", end="2011-01-10", periods=rows)
    df.index = pd.DatetimeIndex(data=vals)
    return df


DATASETS = [
    # _getDataset(0, 1),
    # _getDataset(1, 1),
    _getDataset(100, 1),
    # _getDataset(1000, 1),
    # _getDataset(0, 4),
    # _getDataset(1, 4),
    _getDataset(100, 4),
    # _getDataset(1000, 4),
    # _getDataset(10000, 40),
    # _getDataset(20, 4),
]


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_setFlagger(data, flagger):

    field, *_ = data.columns

    this_flagger = flagger.initFlags(data)
    other_flagger = this_flagger.getFlagger(iloc=slice(None, None, 3)).setFlags(field)
    result_flagger = this_flagger.setFlagger(other_flagger)

    other_flags = other_flagger.getFlags()
    result_flags = result_flagger.getFlags(field)

    assert np.all(result_flagger.getFlags(loc=other_flagger.getFlags().index) == other_flags)

    assert np.all(result_flags[~result_flags.index.isin(other_flags.index)] == flagger.UNFLAGGED)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_setFlaggerColumnsDiff(data, flagger):

    field, *_ = data.columns
    new_field = field + "_new"
    iloc = slice(None, None, 2)

    other_data = data.iloc[iloc]
    other_data.columns = [new_field] + data.columns[1:].to_list()

    this_flagger = flagger.initFlags(data).setFlags(field, flag=flagger.BAD)
    other_flagger = flagger.initFlags(other_data)
    result_flagger = this_flagger.setFlagger(other_flagger)

    assert np.all(result_flagger.getFlags(new_field, loc=other_data.index) == other_flagger.getFlags(new_field))
    assert np.all(result_flagger.getFlags(new_field, loc=data.index) == flagger.UNFLAGGED)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_setFlaggerIndexDiff(data, flagger):

    field, *_ = data.columns
    iloc = slice(None, None, 2)

    other_data = data.iloc[iloc]
    other_data.index = other_data.index + pd.Timedelta(minutes=2, seconds=25)

    this_flagger = flagger.initFlags(data).setFlags(field, flag=flagger.BAD)
    other_flagger = flagger.initFlags(other_data)
    result_flagger = this_flagger.setFlagger(other_flagger)

    assert np.all(result_flagger.getFlags(field, loc=other_data.index) == other_flagger.getFlags(field))
    assert np.all(result_flagger.getFlags(field, loc=data.index) == this_flagger.getFlags(field))


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_initFlags(data, flagger):
    flags = flagger.initFlags(data).getFlags()
    assert isinstance(flags, pd.DataFrame)
    assert len(flags.index) == len(data.index)
    assert len(flags.columns) >= len(data.columns)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_getFlags(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    # df
    flags0 = flagger.getFlags()
    assert isinstance(flags0, pd.DataFrame)
    assert flags0.shape == data.shape
    assert (flags0.columns == data.columns).all()

    for dt in flags0.dtypes:
        assert dt == flagger.dtype

    # series
    flags1 = flagger.getFlags(field)
    assert isinstance(flags1, pd.Series)
    assert flags1.dtype == flagger.dtype
    assert flags1.shape[0] == data.shape[0]
    assert flags1.name in data.columns


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedDataFrame(data, flagger):

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    mask = np.zeros(len(data), dtype=bool)

    df_tests = [
        (flagger.isFlagged(), mask),
        (flagger.setFlags(field).isFlagged(), ~mask),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator=">"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator="<"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator="=="), ~mask,),
    ]
    for flags, expected in df_tests:
        assert np.all(flags[field] == expected)
        assert isinstance(flags, pd.DataFrame)
        assert flags.shape == data.shape
        assert (flags.columns == data.columns).all()
        for dt in flags.dtypes:
            assert is_bool_dtype(dt)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedSeries(data, flagger):

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    mask = np.zeros(len(data), dtype=bool)

    series_tests = [
        (flagger.isFlagged(field), mask),
        (flagger.setFlags(field).isFlagged(field), ~mask),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(field, flag=flagger.GOOD, comparator=">"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(field, flag=flagger.GOOD, comparator="<"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(field, flag=flagger.GOOD, comparator="=="), ~mask,),
    ]
    for flags, expected in series_tests:
        assert np.all(flags == expected)
        assert isinstance(flags, pd.Series)
        assert flags.dtype == bool
        assert flags.shape[0] == data.shape[0]
        assert flags.name in data.columns


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedSeries(data, flagger):

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    fail_tests = [
        {"flag": pd.Series(index=data.index, data=flagger.BAD).astype(flagger.dtype)},
        {"field": ["var1", "var2"]},
    ]
    for args in fail_tests:
        with pytest.raises(ValueError):
            flagger.isFlagged(**args)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_setFlags(data, flagger):
    flagger = flagger.initFlags(data)
    sl = slice("2011-01-02", "2011-01-05")
    field, *_ = data.columns

    base = flagger.getFlags()

    flagger_good = flagger.setFlags(field, flag=flagger.GOOD, loc=sl)
    flags_good = flagger_good.getFlags()
    assert flags_good.shape == base.shape
    assert (flags_good.columns == base.columns).all()
    assert (flags_good.loc[sl, field] == flagger.GOOD).all()

    # overflag works BAD > GOOD
    flagger_bad = flagger_good.setFlags(field, flag=flagger.BAD)
    assert (flagger_bad.getFlags(field) == flagger.BAD).all()

    # overflag doesn't work GOOD < BAD
    flagger_still_bad = flagger_bad.setFlags(field, flag=flagger.GOOD)
    assert (flagger_still_bad.getFlags(field) == flagger.BAD).all()

    # overflag does work with force
    flagger_forced_good = flagger_bad.setFlags(field, flag=flagger.GOOD, force=True)
    assert (flagger_forced_good.getFlags(field) == flagger.GOOD).all()

    with pytest.raises(ValueError):
        flagger.setFlags(field=None, flag=flagger.BAD)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_clearFlags(data, flagger):
    flagger = flagger.initFlags(data)
    sl = slice("2011-01-02", "2011-01-05")
    field, *_ = data.columns

    base = flagger.getFlags()

    flagger = flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(base)

    flagger = flagger.clearFlags(field)
    assert np.sum(flagger.isFlagged(field)) == 0

    flagger = flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(base)

    flagger = flagger.clearFlags(field, loc=sl)
    unflagged = flagger.isFlagged(field, loc=sl)
    assert np.sum(unflagged) == 0
    assert np.sum(flagger.isFlagged(field)) == len(data) - len(unflagged)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_dtype(data, flagger):

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    tests = (
        flagger.getFlags(field).astype(str),
        "TEST",
        55,
    )

    for test in tests:
        with pytest.raises(TypeError):
            flagger = flagger.setFlags(field, flag=test)
        assert flagger.getFlags(field).dtype == flagger.dtype


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER[-1:])
def test_returnCopy(data, flagger):

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    base = flagger.getFlags()

    assert flagger.getFlags() is not base
    assert flagger.isFlagged() is not base
    assert flagger.setFlags(field) is not flagger
    assert flagger.clearFlags(field) is not flagger


LOC_ILOC_FUNCS = ["isFlagged", "getFlags"]


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("flaggerfunc", LOC_ILOC_FUNCS)
def test_loc(data, flagger, flaggerfunc):

    flagger = flagger.initFlags(data)
    sl = slice("2011-01-02", "2011-01-05")
    field, *_ = data.columns

    chunk = data.loc[sl, field]
    d = data.loc[sl]
    if d.empty:
        mask = []
    else:
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


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("flaggerfunc", LOC_ILOC_FUNCS)
def test_iloc(data, flagger, flaggerfunc):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    M = len(data.index) - 1 if len(data.index) > 0 else 0
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


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_classicUseCases(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    # data-mask, same length than flags
    d = data[field]
    mask = d < (d.max() - d.min()) // 2
    flagger = flagger.clearFlags(field)
    flagged = flagger.setFlags(field, loc=mask, flag=flagger.BAD).isFlagged(field)
    assert (flagged == mask).all()

    # some fun with numpy but not same dimensions.. pass indices to iloc
    indices = np.arange(0, len(data))
    mask = indices % 3 == 0
    indices = indices[mask]
    flagger.clearFlags(field)
    flagged = flagger.setFlags(field, iloc=indices, flag=flagger.BAD).isFlagged(field)
    assert (flagged.iloc[indices] == flagged[flagged]).all()
