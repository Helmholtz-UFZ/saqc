#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

from saqc.core import register
from saqc.funcs.curvefit import _fitPolynomial
from saqc.funcs.rolling import _roll

if TYPE_CHECKING:
    from saqc import SaQC


class ResidualsMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def calculatePolynomialResiduals(
        self: "SaQC",
        field: str,
        window: str | int,
        order: int,
        min_periods: int = 0,
        **kwargs,
    ) -> "SaQC":
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
        window :
            The size of the window you want to use for fitting. If an integer is passed,
            the size refers to the number of periods for every fitting window. If an
            offset string is passed, the size refers to the total temporal extension. The
            window will be centered around the vaule-to-be-fitted. For regularly sampled
            timeseries the period number will be casted down to an odd number if even.

        order :
            The degree of the polynomial used for fitting

        min_periods :
            The minimum number of periods, that has to be available in every values
            fitting surrounding for the polynomial fit to be performed. If there are not
            enough values, np.nan gets assigned. Default (0) results in fitting
            regardless of the number of values present (results in overfitting for too
            sparse intervals). To automatically set the minimum number of periods to the
            number of values in an offset defined window size, pass np.nan.
        """
        orig = self._data[field]
        data, _ = _fitPolynomial(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            order=order,
            min_periods=min_periods,
            **kwargs,
        )
        self._data[field] = orig - data[field]
        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def calculateRollingResiduals(
        self: "SaQC",
        field: str,
        window: str | int,
        func: Callable[[pd.Series], np.ndarray] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate the diff of a rolling-window function and the data.

        Note, that the data gets assigned the worst flag present in the original data.

        Parameters
        ----------
        window :
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.

        func : default mean
            Function to roll with.

        min_periods :
            The minimum number of periods to get a valid value

        center :
            If True, center the rolling window.
        """
        orig = self._data[field].copy()
        data, _ = _roll(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            func=func,
            min_periods=min_periods,
            center=center,
            **kwargs,
        )

        # calculate residual
        self._data[field] = orig - data[field]
        return self
