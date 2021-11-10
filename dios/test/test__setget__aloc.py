from .test_setup import *
from pandas.core.dtypes.common import is_scalar

pytestmark = pytest.mark.skip


@pytest.mark.parametrize(
    ("idxer", "exp"), [("a", s1), ("c", s3), ("x", pd.Series(dtype=float))]
)
def test__getitem_aloc_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.aloc[:, idxer]
    assert isinstance(di, pd.Series)
    assert (di == exp).all()


@pytest.mark.parametrize(("idxer", "exp"), [((1, "a"), s1), ((3, "c"), s3)])
def test__getitem_aloc_singleRow_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.aloc[idxer]
    assert is_scalar(di)
    assert di == exp.loc[idxer[0]]


@pytest.mark.parametrize("idxerL", R_LOC_INDEXER)
@pytest.mark.parametrize("idxerR", C_LOC_INDEXER)
def test__getitem__aloc(dios_aligned, idxerL, idxerR):
    di = dios_aligned.copy().aloc[idxerL, idxerR]
    exp = dios_aligned.copy().loc[idxerL, idxerR]
    assert isinstance(di, DictOfSeries)
    assert (di == exp).all(None)


# #############################
# __SETITEM__


@pytest.mark.parametrize(
    ("idxer", "exp"),
    [
        (slice(None), [s1 == s1, s2 == s2, s3 == s3, s4 == s4]),
        (C_BLIST, [s1 == s1, s2 != s2, s3 != s3, s4 == s4]),
    ],
)
def test__setitem_aloc_singleCol(dios_aligned, idxer, exp):
    di = dios_aligned.copy()
    di.aloc[:, idxer] = 99
    for i, c in enumerate(di):
        assert ((di[c] == 99) == exp[i]).all()


VALS = [
    99,
    pd.Series(range(4, 10), index=range(4, 10)),
]


@pytest.mark.parametrize("idxerL", R_LOC_INDEXER)
@pytest.mark.parametrize("idxerR", C_LOC_INDEXER)
@pytest.mark.parametrize("val", VALS)
def test__setitem__aloc(dios_aligned, idxerL, idxerR, val):
    di = dios_aligned.copy()
    di.aloc[idxerL, idxerR] = val
    exp = dios_aligned.copy()
    di.loc[idxerL, idxerR] = val
    assert isinstance(di, DictOfSeries)
    assert (di == exp).all(None)
