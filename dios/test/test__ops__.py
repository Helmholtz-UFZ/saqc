#!/usr/bin/env python
import pytest

from .test_setup import *


__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Centrum für Umweltforschung GmbH - UFC"


@pytest.mark.parametrize("left", diosFromMatr(DATA_ALIGNED))
@pytest.mark.parametrize("right", diosFromMatr(DATA_ALIGNED))
def test__eq__(left, right):
    a, b = left, right
    _test = a == b
    for c in _test:
        for i in _test[c].index:
            res = (_test[c])[i]
            e1 = a[c][i]
            e2 = b[c][i]
            exp = e1 == e2
            assert res == exp


@pytest.mark.filterwarnings("ignore: invalid value encountered in long_scalars")
@pytest.mark.filterwarnings("ignore: divide by zero encountered in long_scalars")
@pytest.mark.parametrize("left", diosFromMatr(DATA_ALIGNED))
@pytest.mark.parametrize("right", diosFromMatr(DATA_ALIGNED))
@pytest.mark.parametrize("op", OP2)
def test__op2__aligningops(left, right, op):
    a, b = left, right
    test = op(a, b)
    for c in test:
        for j in test[c].index:
            exp = op(a[c][j], b[c][j])
            res = test[c][j]
            if not np.isfinite(res):
                print(f"\n\n{a[c][j]} {OP_MAP[op]} {b[c][j]}")
                print(f"\nres: {res}, exp:{exp}, op: {OP_MAP[op]}")
                pytest.skip("test not support non-finite values")
                return
            assert res == exp


@pytest.mark.filterwarnings("ignore: invalid value encountered in long_scalars")
@pytest.mark.filterwarnings("ignore: divide by zero encountered in long_scalars")
@pytest.mark.parametrize("left", diosFromMatr(DATA_UNALIGNED))
@pytest.mark.parametrize("right", diosFromMatr(DATA_UNALIGNED))
@pytest.mark.parametrize("op", OPNOCOMP)
def test__op2__UNaligningops(left, right, op):
    try:
        a, b = left, right
        test = op(a, b)
        for c in test:
            for j in test[c].index:
                exp = op(a[c][j], b[c][j])
                res = test[c][j]
                if not np.isfinite(res):
                    print(f"\n\n{a[c][j]} {OP_MAP[op]} {b[c][j]}")
                    print(f"\nres: {res}, exp:{exp}, op: {OP_MAP[op]}")
                    pytest.skip("test not support non-finite values")
                    return
                assert res == exp
    except ZeroDivisionError:
        pytest.skip("ZeroDivisionError")


@pytest.mark.parametrize("data", diosFromMatr(ALL))
@pytest.mark.parametrize("op", OP1)
def test__op1__(data, op):
    test = op(data)
    res = [entry for col in test for entry in test[col]]
    e = [entry for col in data for entry in data[col]]
    for i in range(len(res)):
        exp = op(e[i])
        assert res[i] == exp
