#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

import dios

from tests.common import TESTFLAGGER, initData


pytestmark = pytest.mark.skip('old flagger tests - rewrite needed')


def _getDataset(rows, cols):
    return initData(cols=cols, rows=rows, start_date="2011-01-01", end_date="2011-01-10")


DATASETS = [
    _getDataset(0, 1),
    _getDataset(1, 1),
    _getDataset(100, 1),
    # _getDataset(1000, 1),
    _getDataset(0, 4),
    _getDataset(1, 4),
    # _getDataset(100, 4),
    # _getDataset(1000, 4),
    # _getDataset(10000, 40),
    _getDataset(20, 4),
]


def check_all_dios_index_length(tocheck, expected):
    for c in tocheck:
        if len(tocheck[c]) != len(expected[c]):
            return False
    return True


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_initFlags(data, flagger):
    """
    test before:
    - None
    """

    newflagger = flagger.initFlags(data)
    assert isinstance(newflagger, type(flagger))
    assert newflagger is not flagger

    flags = newflagger.getFlags()
    assert isinstance(flags, dios.DictOfSeries)

    assert len(flags.columns) >= len(data.columns)
    assert check_all_dios_index_length(flags, data)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_initFlagsWithFlags(data, flagger):
    flags = dios.DictOfSeries(pd.Series(data=flagger.BAD))
    flagger = flagger.initFlags(flags=flags)
    assert (flagger.flags == flags).all(axis=None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_getFlags(data, flagger):
    """
    test before:
    - initFlags()

    we need to check:
    - access all flags -> get a dios
    - access some columns of flags -> get a dios
    - access one column of flags -> get a series
    """

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    # all - dios
    flags0 = flagger.getFlags()
    assert isinstance(flags0, dios.DictOfSeries)
    assert (flags0.columns == data.columns).all()
    assert check_all_dios_index_length(flags0, data)
    for dt in flags0.dtypes:
        assert dt == flagger.dtype

    # some - dios
    if len(data.columns) >= 2:
        cols = data.columns[:2].to_list()
        flags1 = flagger.getFlags(cols)
        assert isinstance(flags1, dios.DictOfSeries)
        assert (flags1.columns == data.columns[:2]).all()
        assert check_all_dios_index_length(flags1, data[cols])
        for dt in flags1.dtypes:
            assert dt == flagger.dtype

    # series
    flags2 = flagger.getFlags(field)
    assert isinstance(flags2, pd.Series)
    assert flags2.dtype == flagger.dtype
    assert flags2.shape[0] == data[field].shape[0]
    # NOTE: need fix in dios see issue #16 (has very low priority)
    # assert flags2.name in data.columns


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_setFlags(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags()
    """
    flagger = flagger.initFlags(data)
    sl = slice("2011-01-02", "2011-01-05")
    field, *_ = data.columns

    base = flagger.getFlags()

    flagger_good = flagger.setFlags(field, flag=flagger.GOOD, loc=sl)
    assert isinstance(flagger_good, type(flagger))
    assert flagger_good is not flagger

    flags_good = flagger_good.getFlags()
    assert len(flags_good[field]) <= len(base[field])
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
def test_sliceFlagger(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags() inside slice()
    """
    sl = slice(None, None, 3)

    flagger = flagger.initFlags(data)
    newflagger = flagger.slice(loc=sl)
    assert isinstance(newflagger, type(flagger))

    newflags = newflagger.getFlags()
    assert (newflags.columns == data.columns).all()
    assert check_all_dios_index_length(newflags, data[sl])


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_sliceFlaggerDrop(data, flagger):
    flagger = flagger.initFlags(data)
    with pytest.raises(TypeError):
        flagger.getFlags(field=data.columns, drop="var")

    field = data.columns[0]
    expected = data.columns.drop(field)

    filtered = flagger.slice(drop=field)
    assert (filtered.getFlags().columns == expected).all(axis=None)
    assert (filtered.getFlags().to_df().index == data[expected].to_df().index).all(axis=None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlagger(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags()
    - setFlags()
    - slice()
    """
    field, *_ = data.columns
    sl = slice(None, None, 3)

    this_flagger = flagger.initFlags(data)
    other_flagger = this_flagger.slice(loc=sl).setFlags(field)
    result_flagger = this_flagger.merge(other_flagger)

    result_flags = result_flagger.getFlags()
    other_flags = other_flagger.getFlags()

    # check flags that was set
    check = result_flags.loc[sl, field] == other_flags[field]
    assert check.all(None)
    # check flags that was not set
    mask = ~result_flags[field].index.isin(other_flags[field].index)
    check = result_flags.loc[mask, field] == result_flagger.UNFLAGGED
    assert check.all(None)

    # check unchanged columns
    cols = data.columns.to_list()
    cols.remove(field)
    check = result_flags[cols] == result_flagger.UNFLAGGED
    assert check.all(None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlaggerColumnsDiff(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags()
    - setFlags()
    - slice()
    - merge()
    """
    field, *_ = data.columns
    new_field = field + "_new"
    sl = slice(None, None, 2)

    other_data = data.loc[sl]
    other_data.columns = [new_field] + data.columns[1:].to_list()
    other_flagger = flagger.initFlags(other_data)

    this_flagger = flagger.initFlags(data).setFlags(field, flag=flagger.BAD)
    result_flagger = this_flagger.merge(other_flagger)

    result_flags = result_flagger.getFlags()
    other_flags = other_flagger.getFlags()

    # we need to check if
    # - the new column is present
    # - the new column is identical to the original
    # - the other column are unchanged
    #   - field-column is BAD
    #   - other columns are UNFLAGGED

    assert new_field in result_flags

    check = result_flags[new_field] == other_flags[new_field]
    assert check.all(None)

    check = result_flags[field] == result_flagger.BAD
    assert check.all(None)

    cols = data.columns.to_list()
    cols.remove(field)
    check = result_flags[cols] == result_flagger.UNFLAGGED
    assert check.all(None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlaggerIndexDiff(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags()
    - setFlags()
    - slice()
    - merge()

    we need to check:
    - index is union of this and other's index
    - indices + values that only in this, should be present
    - indices + values that only in other, should be present
    - indices that in this and other, have values from other
    """
    field, *_ = data.columns
    sl = slice(None, None, 2)

    def shiftindex(s):
        s.index = s.index + pd.Timedelta(minutes=2, seconds=25)
        return s

    # create a sliced time-shifted version of data
    other_data = data.loc[sl].apply(shiftindex)
    if isinstance(other_data, pd.Series):
        pass

    this_flagger = flagger.initFlags(data).setFlags(field, flag=flagger.BAD)
    other_flagger = flagger.initFlags(other_data)
    result_flagger = this_flagger.merge(other_flagger)

    result_flags = result_flagger.getFlags()
    this_flags = this_flagger.getFlags()
    other_flags = other_flagger.getFlags()

    for c in result_flags:
        t, o, r = this_flags[c], other_flags[c], result_flags[c]
        assert (r.index == t.index.union(o.index)).all()

        only_this = t.index.difference(o.index)
        only_other = o.index.difference(t.index)
        both = t.index.intersection(o.index)

        # nothing is missing
        assert (r.index == only_this.union(only_other).union(both)).all()

        assert (r[only_this] == t[only_this]).all()
        assert (r[only_other] == o[only_other]).all()
        assert (r[both] == o[both]).all()


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlaggerOuter(data, flagger):

    field = data.columns[0]

    data_left = data
    data_right = data.iloc[::2]

    left = flagger.initFlags(data=data_left).setFlags(field=field, flag=flagger.BAD)

    right = flagger.initFlags(data=data_right).setFlags(field, flag=flagger.GOOD)

    merged = left.merge(right, join="outer")

    loc = data_right[field].index.difference(data_left[field].index)
    assert (merged.getFlags(field, loc=loc) == flagger.GOOD).all(axis=None)
    assert (merged.getFlags(field, loc=data_left[field].index) == flagger.BAD).all(axis=None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlaggerInner(data, flagger):

    field = data.columns[0]

    data_left = data
    data_right = data.iloc[::2]

    left = flagger.initFlags(data=data_left).setFlags(field=field, flag=flagger.BAD)

    right = flagger.initFlags(data=data_right).setFlags(field, flag=flagger.GOOD)

    merged = left.merge(right, join="inner")

    assert (merged.getFlags(field).index == data_right[field].index).all()
    assert (merged.getFlags(field) == flagger.BAD).all()


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_mergeFlaggerMerge(data, flagger):

    field = data.columns[0]
    data_left = data
    data_right = data.iloc[::2]

    left = flagger.initFlags(data=data_left).setFlags(field=field, flag=flagger.BAD)

    right = flagger.initFlags(data=data_right).setFlags(field, flag=flagger.GOOD)

    merged = left.merge(right, join="merge")

    loc = data_left[field].index.difference(data_right[field].index)
    assert (merged.getFlags(field, loc=data_right[field].index) == flagger.GOOD).all(axis=None)
    assert (merged.getFlags(field, loc=loc) == flagger.BAD).all(axis=None)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedDios(data, flagger):
    """
    test before:
    - initFlags()
    - setFlags()
    """
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    mask = np.zeros(len(data[field]), dtype=bool)

    df_tests = [
        (flagger.isFlagged(), mask),
        (flagger.setFlags(field).isFlagged(), ~mask),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator=">"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator="<"), mask,),
        (flagger.setFlags(field, flag=flagger.GOOD).isFlagged(flag=flagger.GOOD, comparator="=="), ~mask,),
    ]
    for flags, expected in df_tests:
        assert np.all(flags[field] == expected)
        assert isinstance(flags, dios.DictOfSeries)
        assert check_all_dios_index_length(flags, data)
        assert (flags.columns == data.columns).all()
        for dt in flags.dtypes:
            assert is_bool_dtype(dt)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedSeries(data, flagger):
    """
    test before:
    - initFlags()
    - setFlags()
    """
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    mask = np.zeros(len(data[field]), dtype=bool)

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
        assert flags.shape[0] == data[field].shape[0]
        # NOTE: need fix in dios see issue #16 (has very low priority)
        # assert flags.name in data.columns


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_isFlaggedSeries_fail(data, flagger):
    """
    test before:
    - initFlags()
    """
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    fail_tests = [
        {"flag": pd.Series(index=data[field].index, data=flagger.BAD).astype(flagger.dtype)},
        # NOTE: allowed since use of dios
        # {"field": ["var1", "var2"]},
    ]
    for args in fail_tests:
        with pytest.raises(TypeError):
            flagger.isFlagged(**args)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_clearFlags(data, flagger):
    """
    test before:
    - initFlags()
    - getFlags()
    - setFlags()
    - isFlagged()
    """
    flagger = flagger.initFlags(data)
    sl = slice("2011-01-02", "2011-01-05")
    field, *_ = data.columns

    base = flagger.getFlags(field)

    flagger = flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(base)

    flaggernew = flagger.clearFlags(field)
    assert isinstance(flaggernew, type(flagger))
    assert flaggernew is not flagger
    assert len(flagger.getFlags(field)) == len(data[field])

    flagger = flagger.clearFlags(field)
    assert np.sum(flagger.isFlagged(field)) == 0
    assert len(flagger.getFlags(field)) == len(data[field])

    flagger = flagger.setFlags(field=field, flag=flagger.BAD)
    assert np.sum(flagger.isFlagged(field)) == len(base)
    assert len(flagger.getFlags(field)) == len(data[field])

    flagger = flagger.clearFlags(field, loc=sl)
    assert len(flagger.getFlags(field)) == len(data[field])
    unflagged = flagger.isFlagged(field, loc=sl)
    assert np.sum(unflagged) == 0
    assert np.sum(flagger.isFlagged(field)) == len(data[field]) - len(unflagged)


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
        return

    m = data[field].index.get_loc(d[field].index[0])
    M = data[field].index.get_loc(d[field].index[-1])
    mask = np.full(len(data[field]), False)
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
def test_classicUseCases(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    flagger = flagger.clearFlags(field)

    # data-mask, same length than flags
    d = data[field]
    mask = d < (d.max() - d.min()) // 2
    flagged = flagger.setFlags(field, loc=mask, flag=flagger.BAD).isFlagged(field)
    assert (flagged == mask).all()

    flagger = flagger.clearFlags(field)

    indices = np.arange(0, len(data[field]))
    mask = indices % 3 == 0
    indices = indices[mask]
    # we had some fun with numpy and end up with
    # numpy indices (positional), but with different length..
    # make dt-index with iloc, then pass to loc
    dt_idx = data[field].iloc[indices].index
    flagged = flagger.setFlags(field, loc=dt_idx, flag=flagger.BAD).isFlagged(field)
    assert (flagged.iloc[indices] == flagged[flagged]).all()


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_getFlagsWithExtras(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    flags, extra = flagger.getFlags(field, full=True)
    assert isinstance(flags, pd.Series)
    assert isinstance(extra, dict)
    for k, v in extra.items():
        assert isinstance(v, pd.Series)
        assert flags.index.equals(v.index)

    flags, extra = flagger.getFlags(full=True)
    assert isinstance(flags, dios.DictOfSeries)
    assert isinstance(extra, dict)
    for k, v in extra.items():
        assert isinstance(v, dios.DictOfSeries)
        assert flags.columns.equals(v.columns)
        for c in flags:
            assert flags[c].index.equals(v[c].index)


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_replace_delete(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns
    newflagger = flagger.replaceField(field=field, flags=None)

    new, newextra = newflagger.getFlags(full=True)
    assert field not in newflagger.flags
    for k in newextra:
        assert field not in newextra[k]

    with pytest.raises(ValueError):
        flagger.replaceField(field="i_dont_exist", flags=None)

@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_replace_insert(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns
    newfield = 'fooo'
    flags, extra = flagger.getFlags(field, full=True)
    newflagger = flagger.replaceField(field=newfield, flags=flags, **extra)
    old, oldextra = flagger.getFlags(full=True)
    new, newextra = newflagger.getFlags(full=True)
    assert newfield in newflagger.flags
    assert (newflagger._flags[newfield] == flagger._flags[field]).all()
    assert newflagger._flags[newfield] is not flagger._flags[field]  # not a copy
    for k in newextra:
        assert newfield in newextra[k]
        assert (newextra[k][newfield] == oldextra[k][field]).all()


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_replace_replace(data, flagger):
    flagger = flagger.initFlags(data)
    field, *_ = data.columns
    flags, extra = flagger.getFlags(field, full=True)

    # set everything to DOUBTFUL
    flags[:] = flagger.BAD
    for k, v in extra.items():
        v[:] = flagger.BAD
        extra[k] = v

    newflagger = flagger.replaceField(field=field, flags=flags, **extra)

    old, oldextra = flagger.getFlags(full=True)
    new, newextra = newflagger.getFlags(full=True)
    assert old.columns.equals(new.columns)
    assert (new[field] == flagger.BAD).all()

    assert oldextra.keys() == newextra.keys()
    for k in newextra:
        o, n = oldextra[k], newextra[k]
        assert n.columns.equals(o.columns)
        assert (n[field] == flagger.BAD).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagAfter(flagger):
    idx = pd.date_range("2000", "2001", freq='1M')
    s = pd.Series(0, index=idx)
    data = dios.DictOfSeries(s, columns=['a'])
    exp_base = pd.Series(flagger.UNFLAGGED, index=idx)

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    flags = flagger.setFlags(field, loc=s.index[3], flag_after=5).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[3: 3+5+1] = flagger.BAD
    assert (flags == exp).all()

    flags = flagger.setFlags(field, loc=s.index[3], flag_after=5, win_flag=flagger.GOOD).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[3: 3+5+1] = flagger.GOOD
    exp[3] = flagger.BAD
    assert (flags == exp).all()

    # 3 month < 99 days < 4 month
    flags = flagger.setFlags(field, loc=s.index[3], flag_after="99d").getFlags(field)
    exp = exp_base.copy()
    exp.iloc[3: 3+3+1] = flagger.BAD
    assert (flags == exp).all()

    # 3 month < 99 days < 4 month
    flags = flagger.setFlags(field, loc=s.index[3], flag_after="99d", win_flag=flagger.GOOD).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[3: 3+3+1] = flagger.GOOD
    exp[3] = flagger.BAD
    assert (flags == exp).all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_flagBefore(flagger):
    idx = pd.date_range("2000", "2001", freq='1M')
    s = pd.Series(0, index=idx)
    data = dios.DictOfSeries(s, columns=['a'])
    exp_base = pd.Series(flagger.UNFLAGGED, index=idx)

    flagger = flagger.initFlags(data)
    field, *_ = data.columns

    flags = flagger.setFlags(field, loc=s.index[8], flag_before=5).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[8-5: 8+1] = flagger.BAD
    assert (flags == exp).all()

    flags = flagger.setFlags(field, loc=s.index[8], flag_before=5, win_flag=flagger.GOOD).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[8-5: 8+1] = flagger.GOOD
    exp[8] = flagger.BAD
    assert (flags == exp).all()

    # 3 month < 99 days < 4 month
    flags = flagger.setFlags(field, loc=s.index[8], flag_before="99d").getFlags(field)
    exp = exp_base.copy()
    exp.iloc[8-3: 8+1] = flagger.BAD
    assert (flags == exp).all()

    # 3 month < 99 days < 4 month
    flags = flagger.setFlags(field, loc=s.index[8], flag_before="99d", win_flag=flagger.GOOD).getFlags(field)
    exp = exp_base.copy()
    exp.iloc[8-3: 8+1] = flagger.GOOD
    exp[8] = flagger.BAD
    assert (flags == exp).all()
