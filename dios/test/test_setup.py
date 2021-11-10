from dios import *
import pytest
from numpy.random import randint

try:
    from dios.operators import (
        OP_MAP,
        _OP1_MAP,
        _OP2_DIV_MAP,
        _OP2_ARITH_MAP,
        _OP2_BOOL_MAP,
        _OP2_COMP_MAP,
    )
except ModuleNotFoundError:
    from dios.dios.operators import (
        OP_MAP,
        _OP1_MAP,
        _OP2_DIV_MAP,
        _OP2_ARITH_MAP,
        _OP2_BOOL_MAP,
        _OP2_COMP_MAP,
    )

import pandas as pd
import numpy as np

a = pd.Series(range(0, 70, 7), dtype=int)
b = pd.Series(range(5, 15, 1), dtype=int)
c = pd.Series(range(7, 107, 10), dtype=int)
d = pd.Series(range(0, 10, 1), dtype=int)

s1, s2, s3, s4 = a, b, c, d


def df_aligned__():
    return pd.DataFrame(
        dict(
            a=a.copy(),
            b=b.copy(),
            c=c.copy(),
            d=d.copy(),
        )
    )


def dios_aligned__():
    return DictOfSeries(
        dict(
            a=a.copy(),
            b=b.copy(),
            c=c.copy(),
            d=d.copy(),
        )
    )


def dios_unaligned__():
    di = dios_aligned__().copy()
    for i, s in enumerate(di._data):
        s.index = s.index + i * 2
    return di


def df_unaligned__():
    return dios_unaligned__().to_df()


def dios_fuzzy__(nr_cols=None, mincol=0, maxcol=10, itype=None):
    nr_of_cols = nr_cols if nr_cols else randint(mincol, maxcol + 1)

    ns = 10 ** 9
    sec_per_year = 31536000

    ITYPES = [IntItype, FloatItype, DtItype, ObjItype]
    if itype is not None:
        itype = get_itype(itype)
    else:
        itype = ITYPES[randint(0, len(ITYPES))]

    if itype == IntItype:
        f = lambda i: pd.Int64Index(i.astype(int)).unique()
    elif itype == FloatItype:
        f = lambda i: pd.Float64Index(i).unique()
    elif itype == ObjItype:
        f = lambda i: pd.Index(i.astype(int)).unique().astype(str) + "_str"
    else:  # itype == DtItype:
        f = lambda i: pd.to_datetime(i.astype(int) * ns) + pd.Timedelta("30Y")

    di = DictOfSeries(itype=itype)
    for i in range(nr_of_cols):
        start = randint(0, sec_per_year)
        end = start + randint(0, sec_per_year)
        if end > sec_per_year:
            start, end = end - sec_per_year, start

        base = randint(0, 10 + 1)
        exp = randint(1, int(np.log10(end - start + 100)))
        periods = base ** randint(1, exp + 1)
        index = np.linspace(start, end, periods)
        index = f(index)

        arr = randint(0, 10, len(index))
        di[f"c{i}"] = pd.Series(data=arr, dtype=float, index=index)

    return di


@pytest.fixture
def dios_fuzzy():
    return dios_fuzzy__().copy()


@pytest.fixture
def df_aligned():
    return df_aligned__().copy()


@pytest.fixture
def dios_aligned():
    return dios_aligned__().copy()


@pytest.fixture
def df_unaligned():
    return df_unaligned__().copy()


@pytest.fixture
def dios_unaligned():
    return dios_unaligned__().copy()


def diosSeries_eq_dfSeries(
    df_s, di_s, with_msg=False, df_s_name="di_s", di_s_name="df_s"
):
    def fail(msg):
        if with_msg:
            return False, msg
        return False

    assert isinstance(df_s, pd.Series)
    assert isinstance(di_s, pd.Series)

    if df_s.empty and not di_s.empty:
        return fail(
            f"value mismatch: " f"{df_s_name} is missing, but " f"{di_s_name} == {di_s}"
        )

    idiff = di_s.index.difference(df_s.index)
    if not idiff.empty:
        return fail(
            f"index mismatch: "
            f"{di_s_name}.index: {di_s.index.to_list()}, "
            f"{df_s_name}.index: {df_s.index.to_list()}, "
            f"diff: {idiff.to_list()}"
        )

    # compare series
    for i in df_s.index:
        exp = df_s.loc[i]

        # Normally df-nans, from selecting are just not present values
        # in a dios. But if a Nan was inserted in dios on purpose, it is
        # a valid value, so we try to access the value first.
        try:
            val = di_s.loc[i]
        except KeyError:
            # nan in df, missing in dios -> OK
            if np.isnan(exp):
                continue

            # valid val in df, missing in dios -> FAIL
            else:
                return fail(
                    f"value mismatch: "
                    f"{di_s_name}.loc[{i}] == {exp}, but "
                    f"{df_s_name}.loc[{i}] does not exist"
                )

        # inf = np.isinf(exp) and np.isinf(val)
        # sig = np.sign(exp) == np.sign(val)
        # eq_nan = np.isnan(exp) and np.isnan(val)
        # eq_inf = inf and sig
        # eq_vals = exp == val
        # eq = eq_nan or eq_inf or eq_vals
        eq = np.equal(val, exp)
        assert np.isscalar(eq)

        if not eq:
            return fail(
                f"value mismatch: "
                f"{di_s_name}.loc[{i}] == {exp}, but "
                f"{df_s_name}.loc[{i}] == {val}"
            )

    return True, "equal" if with_msg else True


def dios_eq_df(dios, df, dios_dropped_empty_colums=False, with_msg=False):
    def fail(msg):
        if with_msg:
            return False, msg
        return False

    assert isinstance(df, pd.DataFrame)
    assert isinstance(dios, DictOfSeries)

    # check: dios has not more/other cols than df
    notmore = [c for c in dios if c not in df]
    if notmore:
        return fail(
            f"columns mismatch. "
            f"dios: {dios.columns.to_list()}, "
            f"df: {df.columns.to_list()}, "
            f"diff: {notmore}"
        )

    # check: may df has empty cols and dios has no cols
    # at this locations
    miss = [c for c in df if c not in dios]
    if miss:
        if dios_dropped_empty_colums:
            tmp = []
            for c in miss:
                if not df[c].dropna().empty:
                    tmp += [c]
            if tmp:
                return fail(f"columns mismatch: " f"dios missing column(s): {tmp}")
        else:
            return fail(f"columns mismatch: " f"dios missing column(s): {miss}")

    cols = df.columns.intersection(dios.columns)

    for c in cols:
        ok, m = diosSeries_eq_dfSeries(
            df[c], dios[c], di_s_name=f"di[{c}]", df_s_name=f"df[{c}]", with_msg=True
        )
        if not ok:
            return fail(m)

    return True, "equal" if with_msg else True


# 0,1
NICE_SLICE = [slice(None), slice(None, None, 3)]
R_BLIST = [True, False, False, False, True] * 2
C_BLIST = [True, False, False, True]

#              3,4               5       6
R_LOC_SLICE = NICE_SLICE + [slice(2), slice(2, 8)]
R_LOC_LIST = [[1], [3, 4, 5], pd.Series([3, 7])]
#              7            8                  9
R_LOC_BLIST = [R_BLIST, pd.Series(R_BLIST), pd.Series(R_BLIST).values]

#              0,      1,           2,
C_LOC_LIST = [["a"], ["a", "c"], pd.Series(["a", "c"])]
C_LOC_SLICE = NICE_SLICE + [slice("b"), slice("b", "c")]
C_LOC_BLIST = [
    C_BLIST,
    pd.Series(C_BLIST, index=list("abcd")),
    pd.Series(C_BLIST).values,
]

#                 0 1            2           3            4
RC_iLOC_SLICE = NICE_SLICE + [slice(4), slice(-3, -1), slice(-1, 3)]
R_iLOC_LIST = [[7], [6, 8]]
R_iLOC_BLIST = [
    R_BLIST,
    pd.Series(R_BLIST).values,
]  # only list-likes allowed not series-likes
C_iLOC_LIST = [[0], [1, 3]]
C_iLOC_BLIST = [C_BLIST, pd.Series(C_BLIST).values]

MULTIIDXER = [
    df_aligned__() > 9,
    df_aligned__() != df_aligned__(),
    df_aligned__() == df_aligned__(),
    df_aligned__() % 3 == 0,
]
EMPTYIDEXER = [
    [],
    pd.Series(dtype="O"),
]
EMPTY_DF = [pd.DataFrame()]

BASIC_INDEXER = (
    C_LOC_LIST + R_LOC_SLICE + R_LOC_BLIST + MULTIIDXER + EMPTYIDEXER + EMPTY_DF
)
BASIC_INDEXER_FAIL = [
    ["z"],
    ["a", "z"],
    pd.Series(["a", "z"]),
    pd.DataFrame(dict(a=[1, 2, 3])),
]

R_LOC_INDEXER = R_LOC_SLICE + R_LOC_LIST + R_LOC_BLIST + EMPTYIDEXER
C_LOC_INDEXER = C_LOC_SLICE + C_LOC_LIST + C_LOC_BLIST + EMPTYIDEXER

R_iLOC_INDEXER = RC_iLOC_SLICE + R_iLOC_LIST + R_iLOC_BLIST
C_iLOC_INDEXER = RC_iLOC_SLICE + C_iLOC_LIST + C_iLOC_BLIST

O = [[0, 0, 0], [0, 0, 0]]
I = [[1, 1, 1], [1, 1, 1]]
A = [[1, 2, 3], [4, 5, 6]]
B = [[0, 2, 2], [5, 5, 5]]
C = [[3, 2, 0], [1, 0, 3]]
D = [[6, 5, 4], [3, 2, 1]]
DATA_ALIGNED = [O, I, A, B, C, D]

# outer lists could have differnet length, but this would
# make the checks to complicated
EEE = [[], [], []]
O = [[0, 0], [0, 0, 0], [0, 0, 0, 0]]
I = [[1, 1, 1], [1, 1, 1], [1]]
A = [[1], [2, 3], [4, 5, 6]]
B = [[0, 2, 2], [5], [5, 5]]
C = [[3, 2, 0], [1, 0, 3], [0, 0, 0]]
D = [[6], [2], [9]]
DATA_UNALIGNED = [O, I, A, B, C, D, EEE]

# only use if a single matrix is used
ALL = DATA_ALIGNED + DATA_UNALIGNED

OPCOMP = list(_OP2_COMP_MAP)
OPNOCOMP = list(_OP2_ARITH_MAP) + list(_OP2_BOOL_MAP) + list(_OP2_DIV_MAP)
OP2 = OPCOMP + OPNOCOMP
OP1 = list(_OP1_MAP)


def diosFromMatr(mlist):
    l = []
    for m in mlist:
        l.append(DictOfSeries({i: li.copy() for i, li in enumerate(m)}))
    return tuple(l)


@pytest.fixture()
def datetime_series():
    m = randint(2, 1000)
    idx = pd.date_range("2000", "2010", m)
    return pd.Series(range(m), idx)
