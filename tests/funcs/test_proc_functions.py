#! /usr/bin/env python
# -*- coding: utf-8 -*-


# see test/functs/fixtures.py for global fixtures "course_..."

import dios

from saqc.constants import *
from saqc.core import initFlagsLike
from saqc.funcs.transformation import transform
from saqc.funcs.drift import correctOffset
from saqc.funcs.interpolation import (
    interpolateByRolling,
    interpolateInvalid,
    interpolateIndex,
)
from saqc.funcs.resampling import resample
from saqc.lib.ts_operators import linearInterpolation, polynomialInterpolation

from tests.fixtures import *


def test_rollingInterpolateMissing(course_5):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    dataInt, *_ = interpolateByRolling(
        data.copy(),
        field,
        flags.copy(),
        3,
        func=np.median,
        center=True,
        min_periods=0,
        interpol_flag=UNFLAGGED,
    )
    assert dataInt[field][characteristics["missing"]].notna().all()
    dataInt, *_ = interpolateByRolling(
        data.copy(),
        field,
        flags.copy(),
        3,
        func=np.nanmean,
        center=False,
        min_periods=3,
        interpol_flag=UNFLAGGED,
    )
    assert dataInt[field][characteristics["missing"]].isna().all()


def test_interpolateMissing(course_5):
    data, characteristics = course_5(periods=10, nan_slice=[5])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    dataLin, *_ = interpolateInvalid(data, field, flags, method="linear")
    dataPoly, *_ = interpolateInvalid(data, field, flags, method="polynomial")
    assert dataLin[field][characteristics["missing"]].notna().all()
    assert dataPoly[field][characteristics["missing"]].notna().all()
    data, characteristics = course_5(periods=10, nan_slice=[5, 6, 7])
    dataLin1, *_ = interpolateInvalid(
        data.copy(), field, flags, method="linear", limit=2
    )
    dataLin2, *_ = interpolateInvalid(
        data.copy(), field, flags, method="linear", limit=3
    )
    dataLin3, *_ = interpolateInvalid(
        data.copy(), field, flags, method="linear", limit=4
    )
    assert dataLin1[field][characteristics["missing"]].isna().all()
    assert dataLin2[field][characteristics["missing"]].isna().all()
    assert dataLin3[field][characteristics["missing"]].notna().all()


def test_transform(course_5):
    data, characteristics = course_5(periods=10, nan_slice=[5, 6])
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    data1, *_ = transform(data, field, flags, func=linearInterpolation)
    assert data1[field][characteristics["missing"]].isna().all()
    data1, *_ = transform(
        data, field, flags, func=lambda x: linearInterpolation(x, inter_limit=3)
    )
    assert data1[field][characteristics["missing"]].notna().all()
    data1, *_ = transform(
        data,
        field,
        flags,
        func=lambda x: polynomialInterpolation(x, inter_limit=3, inter_order=3),
    )
    assert data1[field][characteristics["missing"]].notna().all()


def test_resample(course_5):
    data, characteristics = course_5(
        freq="1min", periods=30, nan_slice=[1, 11, 12, 22, 24, 26]
    )
    field = data.columns[0]
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    data1, *_ = resample(
        data,
        field,
        flags,
        "10min",
        np.mean,
        maxna=2,
        maxna_group=1,
    )
    assert ~np.isnan(data1[field].iloc[0])
    assert np.isnan(data1[field].iloc[1])
    assert np.isnan(data1[field].iloc[2])


def test_interpolateGrid(course_5, course_3):
    data, _ = course_5()
    data_grid, characteristics = course_3()
    data["grid"] = data_grid.to_df()
    # data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    dataInt, *_ = interpolateIndex(
        data, "data", flags, "1h", "time", grid_field="grid", limit=10
    )


def test_offsetCorrecture():
    data = pd.Series(0, index=pd.date_range("2000", freq="1d", periods=100), name="dat")
    data.iloc[30:40] = -100
    data.iloc[70:80] = 100
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    data, _ = correctOffset(data, "dat", flags, 40, 20, "3d", 1)
    assert (data == 0).all()[0]
