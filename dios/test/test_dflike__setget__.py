from .test_setup import *
import pytest


def _test(res, exp):

    if isinstance(exp, pd.DataFrame):
        eq, msg = dios_eq_df(res, exp, with_msg=True)
        assert eq, msg

    else:
        assert type(exp) == type(res)

        if isinstance(exp, pd.Series):
            eq, msg = diosSeries_eq_dfSeries(res, exp, with_msg=True)
            assert eq, msg

        # scalars
        else:
            assert res == exp


@pytest.mark.parametrize("idxer", BASIC_INDEXER)
def test_dflike__get__(df_aligned, dios_aligned, idxer):
    print(idxer)
    exp = df_aligned[idxer]
    res = dios_aligned[idxer]
    _test(res, exp)


@pytest.mark.parametrize("locR", R_LOC_INDEXER)
@pytest.mark.parametrize("locC", C_LOC_INDEXER)
def test_dflike__get_loc__(df_aligned, dios_aligned, locR, locC):
    print(locR)
    print(locC)
    exp = df_aligned.loc[locR, locC]
    res = dios_aligned.loc[locR, locC]
    _test(res, exp)


@pytest.mark.parametrize("ilocR", R_iLOC_INDEXER)
@pytest.mark.parametrize("ilocC", C_iLOC_INDEXER)
def test_dflike__get_iloc__(df_aligned, dios_aligned, ilocR, ilocC):
    print(ilocR)
    print(ilocC)
    exp = df_aligned.iloc[ilocR, ilocC]
    res = dios_aligned.iloc[ilocR, ilocC]
    _test(res, exp)


VALS = [
    99,
]


@pytest.mark.parametrize("idxer", BASIC_INDEXER)
@pytest.mark.parametrize("val", VALS)
def test_dflike__set__(df_aligned, dios_aligned, idxer, val):
    print(idxer)
    exp = df_aligned
    res = dios_aligned
    # NOTE: two test fail, pandas bul***it
    #   df[:2]    -> select 2 rows
    #   df[:2]=99 -> set 3 rows, WTF ???
    exp[idxer] = val
    res[idxer] = val
    _test(res, exp)


@pytest.mark.parametrize("locR", R_LOC_INDEXER)
@pytest.mark.parametrize("locC", C_LOC_INDEXER)
@pytest.mark.parametrize("val", VALS)
def test_dflike__set_loc__(df_aligned, dios_aligned, locR, locC, val):
    print(locR)
    print(locC)
    exp = df_aligned
    res = dios_aligned
    exp.loc[locR, locC] = val
    res.loc[locR, locC] = val
    _test(res, exp)


@pytest.mark.parametrize("ilocR", R_iLOC_INDEXER)
@pytest.mark.parametrize("ilocC", C_iLOC_INDEXER)
@pytest.mark.parametrize("val", VALS)
def test_dflike__set_iloc__(df_aligned, dios_aligned, ilocR, ilocC, val):
    print(ilocR)
    print(ilocC)
    exp = df_aligned
    res = dios_aligned
    exp.iloc[ilocR, ilocC] = val
    res.iloc[ilocR, ilocC] = val
    _test(res, exp)
