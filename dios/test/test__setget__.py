from .test_setup import *
from pandas.core.dtypes.common import is_scalar


@pytest.mark.parametrize(("idxer", "exp"), [("a", s1), ("c", s3)])
def test__getitem_single(dios_aligned, idxer, exp):
    di = dios_aligned[idxer]
    assert isinstance(di, pd.Series)
    assert (di == exp).all()


@pytest.mark.parametrize(
    "idxer",
    [
        "x",
        "2",
        1000,
        None,
    ],
)
def test__getitem_single_fail(dios_aligned, idxer):
    with pytest.raises((KeyError, ValueError)):
        di = dios_aligned[idxer]


@pytest.mark.parametrize("idxer", BASIC_INDEXER)
def test__getitem_(dios_aligned, idxer):
    di = dios_aligned[idxer]

    assert isinstance(di, DictOfSeries)


@pytest.mark.parametrize("idxer", BASIC_INDEXER_FAIL)
def test__getitem_fail(dios_aligned, idxer):
    with pytest.raises((ValueError, KeyError)):
        dios_aligned[idxer]


@pytest.mark.parametrize(
    ("idxer", "exp"),
    [
        (slice(None), [s1 == s1, s2 == s2, s3 == s3, s4 == s4]),
        (dios_aligned__() > 5, [s1 > 5, s2 > 5, s3 > 5, s4 > 5]),
    ],
)
def test__setitem_single(dios_aligned, idxer, exp):
    di = dios_aligned
    di[idxer] = 99
    for i, c in enumerate(di):
        assert ((di[c] == 99) == exp[i]).all()


@pytest.mark.parametrize("idxer", BASIC_INDEXER_FAIL)
def test__setitem__fail(dios_aligned, idxer):
    with pytest.raises((ValueError, KeyError, IndexError)):
        dios_aligned[idxer] = 99
