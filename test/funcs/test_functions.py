#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags

from saqc.lib.tools import getPandasData

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
    data = pd.DataFrame(data={field: np.ones(index.size) * 50}, index=index)
    flags = flagger.initFlags(data)

    # test
    data, flags = flagSesonalRange(data, flags, field, flagger, min=40, max=60, startmonth=7, startday=1, endmonth=8, endday=31)
    flagged = flagger.isFlagged(flags[field])
    assert len(flags[flagged]) == (31 + 31) * 4

    flags = flagger.initFlags(data)
    data, flags = flagSesonalRange(data, flags, field, flagger, min=40, max=60, startmonth=12, startday=16, endmonth=1, endday=15)
    flagged = flagger.isFlagged(flags[field])
    assert len(flags[flagged]) == 31 * 4


if __name__ == '__main__':
    for f in TESTFLAGGERS:
        test_range(f)
        test_flagSesonalRange(f)
