from .test_setup import *
from pandas.core.dtypes.common import is_scalar


@pytest.mark.parametrize(("idxer", "exp"), [("a", s1), ("c", s3)])
def test__getitem_loc_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.loc[:, idxer]
    assert isinstance(di, pd.Series)
    assert (di == exp).all()


@pytest.mark.parametrize(("idxer", "exp"), [((1, "a"), s1), ((3, "c"), s3)])
def test__getitem_loc_singleRow_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.loc[idxer]
    assert is_scalar(di)
    assert di == exp.loc[idxer[0]]


@pytest.mark.parametrize(
    "idxer",
    [
        "x",
        "2",
        1,
        None,
    ],
)
def test__getitem_loc_singleCol_fail(dios_aligned, idxer):
    with pytest.raises((KeyError, TypeError)):
        di = dios_aligned.loc[:, idxer]


# #############################
# __SETITEM__


@pytest.mark.parametrize(
    ("idxer", "exp"),
    [
        (slice(None), [s1 == s1, s2 == s2, s3 == s3, s4 == s4]),
        (C_BLIST, [s1 == s1, s2 != s2, s3 != s3, s4 == s4]),
    ],
)
def test__setitem_loc_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.copy()
    di.loc[:, idxer] = 99
    for i, c in enumerate(di):
        assert ((di[c] == 99) == exp[i]).all()


VALS = [
    99,
]


@pytest.mark.parametrize("idxerL", R_LOC_INDEXER)
@pytest.mark.parametrize("idxerR", C_LOC_INDEXER)
@pytest.mark.parametrize("val", VALS)
def test__setitem__loc(dios_aligned, idxerL, idxerR, val):
    di = dios_aligned.copy()
    di.loc[idxerL, idxerR] = val
    assert isinstance(di, DictOfSeries)
