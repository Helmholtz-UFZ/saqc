#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_range(flagger):
    # prepare
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-02', periods=100)
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    flags = flagger.initFlags(data)
    # test
    data, flags = flagRange(data, flags, field, flagger, min=10, max=90)
    flagged = flagger.isFlagged(flags[field])
    assert len(flags[flagged]) == 10 + 10


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_flagSesonalRange(flagger):
    # prepare
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2014-12-31', freq='1d')
    d = [(x % 2) * 50 for x in range(index.size)]
    data = pd.DataFrame(data={field: d}, index=index)
    flags = flagger.initFlags(data)

    # test
    kwargs = dict(min=1, max=100, startmonth=7, startday=1, endmonth=8, endday=31)
    data, flags = flagSesonalRange(data, flags, field, flagger, **kwargs)
    flagged = flagger.isFlagged(flags[field])
    assert len(flags[flagged]) == (31 + 31) * 4 / 2

    flags = flagger.initFlags(data)
    kwargs = dict(min=1, max=100, startmonth=12, startday=16, endmonth=1, endday=15)
    _, flags = flagSesonalRange(data, flags, field, flagger, **kwargs)
    flagged = flagger.isFlagged(flags[field])
    assert len(flags[flagged]) == 31 * 4 / 2


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_clearFlags(flagger):
    # prepare
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-10', freq='1d')
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    orig = flagger.initFlags(data)
    flags = orig.copy()
    # test
    flags[field] = flagger.setFlag(flags)
    assert (orig != flags).all
    _, cleared = clearFlags(data, flags, field, flagger)
    assert (orig == cleared).all


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_forceFlags(flagger):
    # prepare
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-10', freq='1d')
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    flags = flagger.initFlags(data)
    flags[field] = flagger.setFlag(flags)
    orig = flags.copy()
    # test
    _, foreced = forceFlags(data, flags, field, flagger, flag=flagger.GOOD)
    assert (orig != foreced).all


if __name__ == '__main__':
    for f in TESTFLAGGERS:
        test_range(f)
        test_flagSesonalRange(f)
        test_clearFlags(f)
        test_forceFlags(f)
