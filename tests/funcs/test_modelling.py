#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest

import dios
from saqc import BAD, UNFLAGGED
from saqc.core import initFlagsLike
from saqc.funcs.residuals import calculatePolynomialResiduals, calculateRollingResiduals
from saqc.funcs.tools import selectTime
from tests.fixtures import char_dict, course_1, course_2


@pytest.mark.filterwarnings("ignore: The fit may be poorly conditioned")
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_polyFit_forRegular(dat):
    data, _ = dat(
        freq="10min", periods=30, initial_level=0, final_level=100, out_val=-100
    )
    # add some nice sine distortion
    data = data + 10 * np.sin(np.arange(0, len(data.indexes[0])))
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    result1, _ = calculatePolynomialResiduals(data, "data", flags, 11, 2, numba=False)
    result2, _ = calculatePolynomialResiduals(data, "data", flags, 11, 2, numba=True)
    assert (result1["data"] - result2["data"]).abs().max() < 10**-10
    result3, _ = calculatePolynomialResiduals(
        data, "data", flags, "110min", 2, numba=False
    )
    assert result3["data"].equals(result1["data"])
    result4, _ = calculatePolynomialResiduals(
        data, "data", flags, 11, 2, numba=True, min_periods=11
    )
    assert (result4["data"] - result2["data"]).abs().max() < 10**-10
    data.iloc[13:16] = np.nan
    result5, _ = calculatePolynomialResiduals(
        data, "data", flags, 11, 2, numba=True, min_periods=9
    )
    assert result5["data"].iloc[10:19].isna().all()


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_rollingMean_forRegular(dat):
    data, _ = dat(
        freq="10min", periods=30, initial_level=0, final_level=100, out_val=-100
    )
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    calculateRollingResiduals(
        data,
        "data",
        flags,
        5,
        func=np.mean,
        min_periods=0,
        center=True,
    )
    calculateRollingResiduals(
        data,
        "data",
        flags,
        5,
        func=np.mean,
        min_periods=0,
        center=False,
    )


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_1")])
def test_modelling_mask(dat):
    data, _ = dat()
    data = dios.DictOfSeries(data)
    flags = initFlagsLike(data)
    field = "data"

    # set flags everywhere to test unflagging
    flags[:, field] = BAD

    common = dict(data=data, field=field, flags=flags, mode="periodic")
    data_seasonal, flags_seasonal = selectTime(
        **common, start="20:00", end="40:00", closed=False
    )
    flagscol = flags_seasonal[field]
    m = (20 > flagscol.index.minute) | (flagscol.index.minute > 40)
    assert all(flags_seasonal[field][m] == UNFLAGGED)
    assert all(data_seasonal[field][m].isna())

    data_seasonal, flags_seasonal = selectTime(
        **common, start="15:00:00", end="02:00:00"
    )
    flagscol = flags_seasonal[field]
    m = (15 <= flagscol.index.hour) & (flagscol.index.hour <= 2)
    assert all(flags_seasonal[field][m] == UNFLAGGED)
    assert all(data_seasonal[field][m].isna())

    data_seasonal, flags_seasonal = selectTime(
        **common, start="03T00:00:00", end="10T00:00:00"
    )
    flagscol = flags_seasonal[field]
    m = (3 <= flagscol.index.hour) & (flagscol.index.hour <= 10)
    assert all(flags_seasonal[field][m] == UNFLAGGED)
    assert all(data_seasonal[field][m].isna())

    mask_ser = pd.Series(False, index=data["data"].index)
    mask_ser[::5] = True
    data["mask_ser"] = mask_ser
    flags = initFlagsLike(data)
    data_masked, flags_masked = selectTime(
        data, "data", flags, mode="selection_field", selection_field="mask_ser"
    )
    m = mask_ser
    assert all(flags_masked[field][m] == UNFLAGGED)
    assert all(data_masked[field][m].isna())
