#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from dios import dios
from test.common import TESTFLAGGER

from saqc.funcs.data_modelling import (
    modelling_polyFit,
    modelling_rollingMean
)

TF = TESTFLAGGER[:1]
@pytest.mark.parametrize("flagger", TF)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_polyFit_forRegular(dat, flagger):
    data, _ = dat(freq='10min', periods=30, initial_level=0, final_level=100, out_val=-100)
    # add some nice sine distortion
    data = data + 10*np.sin(np.arange(0, len(data.indexes[0])))
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    result1, _ = modelling_polyFit(data, 'data', flagger, 11, 2, numba=False)
    result2, _ = modelling_polyFit(data, 'data', flagger, 11, 2, numba=True)
    assert (result1['data']-result2['data']).abs().max() < 10**-10
    result3, _ = modelling_polyFit(data, 'data', flagger, '110min', 2, numba=False)
    assert result3['data'].equals(result1['data'])
    result4, _ = modelling_polyFit(data, 'data', flagger, 11, 2, numba=True, min_periods=11)
    assert (result4['data']-result2['data']).abs().max() < 10**-10
    data.iloc[13:16] = np.nan
    result5, _ = modelling_polyFit(data, 'data', flagger, 11, 2, numba=True, min_periods=9)
    assert result5['data'].iloc[10:19].isna().all()

@pytest.mark.parametrize("flagger", TF)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_rollingMean_forRegular(dat, flagger):
    data, _ = dat(freq='10min', periods=30, initial_level=0, final_level=100, out_val=-100)
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    modelling_rollingMean(data, 'data', flagger, 5, eval_flags=True, min_periods=0, center=True)
    modelling_rollingMean(data, 'data', flagger, 5, eval_flags=True, min_periods=0, center=False)
