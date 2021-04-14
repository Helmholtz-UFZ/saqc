from .test_setup import *
from pandas.core.dtypes.common import is_scalar


@pytest.mark.parametrize(
    ("idxer", "exp"),
    [(0, s1), (1, s2), (2, s3), (3, s4), (-1, s4), (-2, s3), (-3, s2), (-4, s1)],
)
def test__getitem_single_iloc(dios_aligned, idxer, exp):
    di = dios_aligned.iloc[:, idxer]
    assert isinstance(di, pd.Series)
    assert (di == exp).all()


@pytest.mark.parametrize(
    ("idxer", "exp"), [((1, 0), s1), ((3, -2), s3), ((-1, -1), s4)]
)
def test__getitem_scalar_iloc(dios_aligned, idxer, exp):
    di = dios_aligned.iloc[idxer]
    assert is_scalar(di)
    assert di == exp.iloc[idxer[0]]


@pytest.mark.parametrize(
    "idxer",
    [
        -5,
        99,
        "a",
        "2",
        None,
    ],
)
def test__getitem_single_iloc_fail(dios_aligned, idxer):
    with pytest.raises((KeyError, IndexError, TypeError)):
        di = dios_aligned.iloc[:, idxer]


# #############################
# __SETITEM__
