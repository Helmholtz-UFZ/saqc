#! /usr/bin/env python
# -*- coding: utf-8 -*-


# see test/functs/fixtures.py for global fixtures "course_..."

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.funcs.tools import mask
from saqc.funcs.residues import calculatePolynomialResidues, calculateRollingResidues

from test.fixtures import *
from test.common import TESTFLAGGER

TF = TESTFLAGGER[:1]


@pytest.mark.parametrize("flagger", TF)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_polyFit_forRegular(dat, flagger):
    data, _ = dat(freq="10min", periods=30, initial_level=0, final_level=100, out_val=-100)
    # add some nice sine distortion
    data = data + 10 * np.sin(np.arange(0, len(data.indexes[0])))
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    result1, _ = calculatePolynomialResidues(data, "data", flagger, 11, 2, numba=False)
    result2, _ = calculatePolynomialResidues(data, "data", flagger, 11, 2, numba=True)
    assert (result1["data"] - result2["data"]).abs().max() < 10 ** -10
    result3, _ = calculatePolynomialResidues(data, "data", flagger, "110min", 2, numba=False)
    assert result3["data"].equals(result1["data"])
    result4, _ = calculatePolynomialResidues(data, "data", flagger, 11, 2, numba=True, min_periods=11)
    assert (result4["data"] - result2["data"]).abs().max() < 10 ** -10
    data.iloc[13:16] = np.nan
    result5, _ = calculatePolynomialResidues(data, "data", flagger, 11, 2, numba=True, min_periods=9)
    assert result5["data"].iloc[10:19].isna().all()


@pytest.mark.parametrize("flagger", TF)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_rollingMean_forRegular(dat, flagger):
    data, _ = dat(freq="10min", periods=30, initial_level=0, final_level=100, out_val=-100)
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    calculateRollingResidues(data, "data", flagger, 5, func=np.mean, eval_flags=True, min_periods=0, center=True)
    calculateRollingResidues(data, "data", flagger, 5, func=np.mean, eval_flags=True, min_periods=0, center=False)


@pytest.mark.parametrize("flagger", TF)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_modelling_mask(dat, flagger):
    data, _ = dat()
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)
    data_seasonal, flagger_seasonal = mask(data, "data", flagger, mode='periodic', period_start="20:00",
                                           period_end="40:00", include_bounds=False)
    flaggs = flagger_seasonal._flags["data"]
    assert flaggs[np.logical_and(20 <= flaggs.index.minute, 40 >= flaggs.index.minute)].isna().all()
    data_seasonal, flagger_seasonal = mask(data, "data", flagger, mode='periodic', period_start="15:00:00",
                                           period_end="02:00:00")
    flaggs = flagger_seasonal._flags["data"]
    assert flaggs[np.logical_and(15 <= flaggs.index.hour, 2 >= flaggs.index.hour)].isna().all()
    data_seasonal, flagger_seasonal = mask(data, "data", flagger, mode='periodic', period_start="03T00:00:00",
                                           period_end="10T00:00:00")
    flaggs = flagger_seasonal._flags["data"]
    assert flaggs[np.logical_and(3 <= flaggs.index.hour, 10 >= flaggs.index.hour)].isna().all()

    mask_ser = pd.Series(False, index=data["data"].index)
    mask_ser[::5] = True
    data["mask_ser"] = mask_ser
    flagger = flagger.initFlags(data)
    data_masked, flagger_masked = mask(data, "data", flagger, mode='mask_var', mask_var="mask_ser")
    flaggs = flagger_masked._flags["data"]
    assert flaggs[data_masked['mask_ser']].isna().all()
