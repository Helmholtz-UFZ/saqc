#! /usr/bin/env python
# -*- coding: utf-8 -*-


# see test/functs/conftest.py for global fixtures "course_..."

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.funcs.proc_functions import (
    proc_interpolateMissing,
    proc_resample,
    proc_transform,
    proc_rollingInterpolateMissing,
    proc_interpolateGrid
)
from saqc.lib.ts_operators import linearInterpolation, polynomialInterpolation

from test.common import TESTFLAGGER

@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_rollingInterpolateMissing(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataInt, *_ = proc_rollingInterpolateMissing(data, field, flagger, 3, func=np.median, center=True,
                                                 min_periods=0, interpol_flag='UNFLAGGED')
    #import pdb
    #pdb.set_trace()
    assert dataInt[field][characteristics['missing']].notna().all()
    dataInt, *_ = proc_rollingInterpolateMissing(data, field, flagger, 3, func=np.nanmean, center=False,
                                                 min_periods=3, interpol_flag='UNFLAGGED')
    assert dataInt[field][characteristics['missing']].isna().all()

@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_interpolateMissing(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataLin, *_ = proc_interpolateMissing(data, field, flagger, method='linear')
    dataPoly, *_ = proc_interpolateMissing(data, field, flagger, method='polynomial')
    assert dataLin[field][characteristics['missing']].notna().all()
    assert dataPoly[field][characteristics['missing']].notna().all()
    data, characteristics = course_5(periods=10, nan_slice=[5, 6, 7])
    dataLin1, *_ = proc_interpolateMissing(data, field, flagger, method='linear', inter_limit=2)
    dataLin2, *_ = proc_interpolateMissing(data, field, flagger, method='linear', inter_limit=3)
    dataLin3, *_ = proc_interpolateMissing(data, field, flagger, method='linear', inter_limit=4)
    assert dataLin1[field][characteristics['missing']].isna().all()
    assert dataLin2[field][characteristics['missing']].isna().all()
    assert dataLin3[field][characteristics['missing']].notna().all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_transform(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data1, *_ = proc_transform(data, field, flagger, func=linearInterpolation)
    assert data1[field][characteristics['missing']].isna().all()
    data1, *_ = proc_transform(data, field, flagger, func=lambda x: linearInterpolation(x, inter_limit=3))
    assert data1[field][characteristics['missing']].notna().all()
    data1, *_ = proc_transform(data, field, flagger, func=lambda x: polynomialInterpolation(x, inter_limit=3,
                                                                                            inter_order=3))
    assert data1[field][characteristics['missing']].notna().all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_resample(course_5, flagger):
    data, characteristics = course_5(freq='1min', periods=30, nan_slice=[1, 11, 12, 22, 24, 26])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data1, *_ = proc_resample(data, field, flagger, '10min', np.mean, max_invalid_total_d=2, max_invalid_consec_d=1)
    assert ~np.isnan(data1[field].iloc[0])
    assert np.isnan(data1[field].iloc[1])
    assert np.isnan(data1[field].iloc[2])

@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_interpolateGrid(course_5, course_3, flagger):
    data, _ = course_5()
    data_grid, characteristics = course_3()
    data['grid']=data_grid.to_df()
    #data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataInt, *_ = proc_interpolateGrid(data, 'data', flagger, '1h', 'time', grid_field='grid', inter_limit=10)

