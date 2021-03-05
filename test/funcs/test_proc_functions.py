#! /usr/bin/env python
# -*- coding: utf-8 -*-


# see test/functs/conftest.py for global fixtures "course_..."

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.funcs.transformation import (
    transform
)
from saqc.funcs.drift import correctOffset
from saqc.funcs.interpolation import interpolateByRolling, interpolateInvalid, interpolateIndex
from saqc.funcs.resampling import resample
from saqc.lib.ts_operators import linearInterpolation, polynomialInterpolation
from saqc.common import *

from test.common import TESTFLAGGER


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_rollingInterpolateMissing(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataInt, *_ = interpolateByRolling(
        data, field, flagger, 3, func=np.median, center=True, min_periods=0, interpol_flag=UNFLAGGED
    )
    # import pdb
    # pdb.set_trace()
    assert dataInt[field][characteristics["missing"]].notna().all()
    dataInt, *_ = interpolateByRolling(
        data, field, flagger, 3, func=np.nanmean, center=False, min_periods=3, interpol_flag=UNFLAGGED
    )
    assert dataInt[field][characteristics["missing"]].isna().all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_interpolateMissing(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataLin, *_ = interpolateInvalid(data, field, flagger, method="linear")
    dataPoly, *_ = interpolateInvalid(data, field, flagger, method="polynomial")
    assert dataLin[field][characteristics["missing"]].notna().all()
    assert dataPoly[field][characteristics["missing"]].notna().all()
    data, characteristics = course_5(periods=10, nan_slice=[5, 6, 7])
    dataLin1, *_ = interpolateInvalid(data, field, flagger, method="linear", inter_limit=2)
    dataLin2, *_ = interpolateInvalid(data, field, flagger, method="linear", inter_limit=3)
    dataLin3, *_ = interpolateInvalid(data, field, flagger, method="linear", inter_limit=4)
    assert dataLin1[field][characteristics["missing"]].isna().all()
    assert dataLin2[field][characteristics["missing"]].isna().all()
    assert dataLin3[field][characteristics["missing"]].notna().all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_transform(course_5, flagger):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data1, *_ = transform(data, field, flagger, func=linearInterpolation)
    assert data1[field][characteristics["missing"]].isna().all()
    data1, *_ = transform(data, field, flagger, func=lambda x: linearInterpolation(x, inter_limit=3))
    assert data1[field][characteristics["missing"]].notna().all()
    data1, *_ = transform(
        data, field, flagger, func=lambda x: polynomialInterpolation(x, inter_limit=3, inter_order=3)
    )
    assert data1[field][characteristics["missing"]].notna().all()


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_resample(course_5, flagger):
    data, characteristics = course_5(freq="1min", periods=30, nan_slice=[1, 11, 12, 22, 24, 26])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data1, *_ = resample(data, field, flagger, "10min", np.mean, max_invalid_total_d=2, max_invalid_consec_d=1)
    assert ~np.isnan(data1[field].iloc[0])
    assert np.isnan(data1[field].iloc[1])
    assert np.isnan(data1[field].iloc[2])


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_interpolateGrid(course_5, course_3, flagger):
    data, _ = course_5()
    data_grid, characteristics = course_3()
    data['grid'] = data_grid.to_df()
    # data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    dataInt, *_ = interpolateIndex(data, 'data', flagger, '1h', 'time', grid_field='grid', inter_limit=10)


@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_offsetCorrecture(flagger):
    data = pd.Series(0, index=pd.date_range('2000', freq='1d', periods=100), name='dat')
    data.iloc[30:40] = -100
    data.iloc[70:80] = 100
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data, flagger = correctOffset(data, 'dat', flagger, 40, 20, '3d', 1)
    assert (data == 0).all()[0]

