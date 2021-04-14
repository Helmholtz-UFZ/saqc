from .test_setup import *


def test_copy_copy_empty(dios_aligned):
    di = dios_aligned
    shallow = di.copy(deep=False)
    deep = di.copy(deep=True)
    empty_w_cols = di.copy_empty(columns=True)
    empty_no_cols = di.copy_empty(columns=False)

    assert di is not shallow
    assert di is not deep
    assert di is not empty_w_cols
    assert di is not empty_no_cols

    for attr in [
        "itype",
        "_itype",
        "_policy",
    ]:
        dios_attr = getattr(di, attr)
        for cop in [shallow, deep, empty_w_cols, empty_no_cols]:
            copy_attr = getattr(cop, attr)
            assert dios_attr == copy_attr

    assert di.columns.equals(shallow.columns)
    assert di.columns.equals(deep.columns)
    assert di.columns.equals(empty_w_cols.columns)
    assert not di.columns.equals(empty_no_cols.columns)

    for i in di:
        assert di._data[i].index is shallow._data[i].index
        assert di._data[i].index is not deep._data[i].index
        di._data[i][0] = 999999
        assert di[i][0] == shallow[i][0]
        assert di[i][0] != deep[i][0]


@pytest.mark.parametrize("left", diosFromMatr(DATA_UNALIGNED))
# we use comp ops just to get some noise in the data
@pytest.mark.parametrize("op", OPCOMP)
def test_all(left, op):
    a = left
    ser = (op(a, a)).all()
    assert isinstance(ser, pd.Series)
    res = [e for e in ser]
    exp = [op(a[col], a[col]) for col in a]
    for i in range(len(res)):
        assert isinstance(exp[i], pd.Series)
        assert (res[i] == exp[i]).all()
