#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dios import DictOfSeries
from saqc.core.flags import Flags
from saqc.core.register import register
from saqc.funcs.curvefit import _fitPolynomial
from saqc.funcs.rolling import _roll


@register(mask=["field"], demask=[], squeeze=[])
def calculatePolynomialResiduals(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    order: int,
    min_periods: Optional[int] = 0,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Fits a polynomial model to the data and calculate the residuals.

    The residual  is calculated by fitting a polynomial of degree `order` to a data
    slice of size `window`, that has x at its center.

    Note, that calculating the residuals tends to be quite costy, because a function
    fitting is performed for every sample. To improve performance, consider the
    following possibilities:

    In case your data is sampled at an equidistant frequency grid:

    (1) If you know your data to have no significant number of missing values,
    or if you do not want to calculate residuals for windows containing missing values
    any way, performance can be increased by setting min_periods=window.

    Note, that the initial and final window/2 values do not get fitted.

    Each residual gets assigned the worst flag present in the interval of
    the original data.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data.

    field : str
        The column, holding the data-to-be-modelled.

    flags : saqc.Flags
        Container to store quality flags to data.

    window : {str, int}
        The size of the window you want to use for fitting. If an integer is passed,
        the size refers to the number of periods for every fitting window. If an
        offset string is passed, the size refers to the total temporal extension. The
        window will be centered around the vaule-to-be-fitted. For regularly sampled
        timeseries the period number will be casted down to an odd number if even.

    order : int
        The degree of the polynomial used for fitting

    min_periods : int or None, default 0
        The minimum number of periods, that has to be available in every values
        fitting surrounding for the polynomial fit to be performed. If there are not
        enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too
        sparse intervals). To automatically set the minimum number of periods to the
        number of values in an offset defined window size, pass np.nan.

    Returns
    -------
    data : dios.DictOfSeries
    flags : saqc.Flags
    """
    orig = data[field]
    data, flags = _fitPolynomial(
        data=data,
        field=field,
        flags=flags,
        window=window,
        order=order,
        min_periods=min_periods,
        **kwargs,
    )
    data[field] = orig - data[field]
    return data, flags


@register(mask=["field"], demask=[], squeeze=[])
def calculateRollingResiduals(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    func: Callable[[pd.Series], np.ndarray] = np.mean,
    min_periods: Optional[int] = 0,
    center: bool = True,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Calculate the diff of a rolling-window function and the data.

    Note, that the data gets assigned the worst flag present in the original data.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data.
    field : str
        The column to calculate on.
    flags : saqc.Flags
        Container to store quality flags to data.
    window : {int, str}
        The size of the window you want to roll with. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string
        is passed, the size refers to the total temporal extension. For regularly
        sampled timeseries, the period number will be casted down to an odd number if
        ``center=True``.
    func : Callable, default np.mean
        Function to roll with.
    min_periods : int, default 0
        The minimum number of periods to get a valid value
    center : bool, default True
        If True, center the rolling window.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    orig = data[field].copy()
    data, flags = _roll(
        data=data,
        field=field,
        flags=flags,
        window=window,
        func=func,
        min_periods=min_periods,
        center=center,
        **kwargs,
    )

    # calculate residual
    data[field] = orig - data[field]
    return data, flags
