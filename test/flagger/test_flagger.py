#!/usr/bin/env python

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger

from pandas.core.indexing import IndexingError

from saqc.funcs.functions import flagRange, flagSesonalRange, forceFlags, clearFlags

TESTFLAGGERS = [
    BaseFlagger(['NIL', 'GOOD', 'BAD']),
    DmpFlagger(),
    SimpleFlagger()]


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_initFlags(flagger):
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-02', periods=100)
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    flags = flagger.initFlags(data)

    assert len(flags) == 100
    assert isinstance(flags, pd.DataFrame)


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_getsetFlags(flagger):
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-02', periods=100)
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    flags = flagger.initFlags(data)

    flags = flagger.setFlags(flags, field, flag=flagger.GOOD)
    flagged = flagger.getFlags(flags)[field]
    assert isinstance(flagged.dtype, pd.CategoricalDtype)
    assert (flagged == flagger.GOOD).all()

    flags = flagger.setFlags(flags, field, flag=flagger.BAD)
    flagged = flagger.getFlags(flags)[field]
    assert (flagged == flagger.BAD).all()

    flags = flagger.setFlags(flags, field, flag=flagger.GOOD)
    flagged = flagger.getFlags(flags)[field]
    assert (flagged == flagger.BAD).all()


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
def test_setFlags_isFlagged(flagger, **kwargs):
    field = 'testdata'
    index = pd.date_range(start='2011-01-01', end='2011-01-02', periods=100)
    data = pd.DataFrame(data={field: np.linspace(0, index.size - 1, index.size)}, index=index)
    flags = flagger.initFlags(data)

    d = data[field]
    mask = d < (d.max() - d.min()) // 2
    assert len(mask) == len(flags.index)

    f = flagger.setFlags(flags, field, loc=mask.values, flag=flagger.BAD)
    # test isFlagged
    isflagged = flagger.isFlagged(f[field])
    assert (isflagged == mask).all()

    # test setFlag with mask
    flagged = flagger.getFlags(f[field])
    isflagged = flagged == flagger.BAD
    assert (isflagged == mask).all()

    # ok we can use isFlagged now :D

    # test with mask and iloc
    f = flagger.setFlags(flags, field, iloc=mask.values, flag=flagger.BAD)
    isflagged = flagger.isFlagged(f[field])
    assert (isflagged == mask).all()

    try:
        m = mask[mask]
        m.iloc[0:10] = False
        m = m[m]
        f = flagger.setFlags(flags, field, loc=m, flag=flagger.BAD)
    except IndexingError:
        pass
    else:
        raise AssertionError

    # test setFlags with loc and index
    idx = mask[mask].index
    assert len(idx) < len(flags.index)
    f = flagger.setFlags(flags, field, loc=idx, flag=flagger.BAD)
    isflagged = flagger.isFlagged(f[field])
    assert (isflagged == mask).all()

    # test setFlags with iloc and index
    idx = mask[mask].reset_index(drop=True).index
    assert len(idx) < len(flags.index)
    f = flagger.setFlags(flags, field, iloc=idx, flag=flagger.BAD)
    isflagged = flagger.isFlagged(f[field])
    assert (isflagged == mask).all()

    # test passing a series of flags as flag-arg
    every = 5
    flagseries = pd.Series(data=flagger.GOOD, index=flags.index)
    flagseries.iloc[::every] = flagger.BAD
    flagseries = flagseries.astype(flagger.flags)
    idx = mask[mask].index
    assert len(flags) == len(flagseries)
    assert len(flags) != len(idx)
    f = flagger.setFlags(flags, field, loc=idx, flag=flagseries)
    bads = flagger.isFlagged(f[field], flag=flagger.BAD, comparator='==')
    bads = bads[bads]
    valid = mask[mask].iloc[::every]
    assert len(valid) == len(bads) and (valid == bads).all()

    # test passing a series of flags as flag-arg and force
    f = flagger.setFlags(flags, field, flag=flagger.BAD)
    every = 5
    flagseries = pd.Series(data=flagger.GOOD, index=flags.index)
    flagseries.iloc[::every] = flagger.UNFLAGGED
    flagseries = flagseries.astype(flagger.flags)
    idx = mask[mask].index
    assert len(flags) == len(flagseries)
    assert len(flags) != len(idx)
    f = flagger.setFlags(f, field, loc=idx, flag=flagseries, force=True)
    unflagged = flagger.isFlagged(f[field], flag=flagger.UNFLAGGED, comparator='==')
    unflagged = unflagged[unflagged]
    valid = mask[mask].iloc[::every]
    assert len(valid) == len(unflagged) and (valid == unflagged).all()




if __name__ == '__main__':
    flagger = TESTFLAGGERS[0]
    test_setFlags_isFlagged(flagger)
    print('done')







