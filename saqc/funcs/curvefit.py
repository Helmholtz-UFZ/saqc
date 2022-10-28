#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from dios import DictOfSeries
from saqc.core.flags import Flags
from saqc.core.register import register
from saqc.lib.tools import getFreqDelta
from saqc.lib.ts_operators import (
    butterFilter,
    polyRoller,
    polyRollerIrregular,
    polyRollerNoMissing,
    polyRollerNoMissingNumba,
    polyRollerNumba,
)

if TYPE_CHECKING:
    from saqc.core.core import SaQC

_FILL_METHODS = Literal[
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
]


class CurvefitMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def fitPolynomial(
        self: "SaQC",
        field: str,
        window: int | str,
        order: int,
        min_periods: int = 0,
        **kwargs,
    ) -> "SaQC":
        """
        Fits a polynomial model to the data.

        The fit is calculated by fitting a polynomial of degree `order` to a data slice
        of size `window`, that has x at its center.

        Note that the result is stored in `field` and overwrite it unless a
        `target` is given.

        In case your data is sampled at an equidistant frequency grid:

        (1) If you know your data to have no significant number of missing values,
        or if you do not want to calculate residuals for windows containing missing values
        any way, performance can be increased by setting min_periods=window.

        Note, that the initial and final window/2 values do not get fitted.

        Each residual gets assigned the worst flag present in the interval of
        the original data.

        Parameters
        ----------
        field : str
             A column in flags and data.

        window : str, int
            Size of the window you want to use for fitting. If an integer is passed,
            the size refers to the number of periods for every fitting window. If an
            offset string is passed, the size refers to the total temporal extension. The
            window will be centered around the vaule-to-be-fitted. For regularly sampled
            data always a odd number of periods will be used for the fit (periods-1 if
            periods is even).

        order : int
            Degree of the polynomial used for fitting

        min_periods : int or None, default 0
            Minimum number of observations in a window required to perform the fit,
            otherwise NaNs will be assigned.
            If ``None``, `min_periods` defaults to 1 for integer windows and to the
            size of the window for offset based windows.
            Passing 0, disables the feature and will result in over-fitting for too
            sparse windows.

        Returns
        -------
        saqc.SaQC
        """
        self._data, self._flags = _fitPolynomial(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            order=order,
            min_periods=min_periods,
            **kwargs,
        )
        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def fitLowpassFilter(
        self: "SaQC",
        field: str,
        cutoff: float | str,
        nyq: float = 0.5,
        filter_order: int = 2,
        fill_method: _FILL_METHODS = "linear",
        **kwargs,
    ):
        """
        Fits the data using the butterworth filter.

        Note
        ----
        The data is expected to be regularly sampled.

        Parameters
        ----------
        field: str
            A column in flags and data.

        cutoff: {float, str}
            The cutoff-frequency, either an offset freq string, or expressed in multiples of the sampling rate.

        nyq: float
            The niquist-frequency. expressed in multiples if the sampling rate.

        fill_method: Literal[‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’]
            Fill method to be applied on the data before filtering (butterfilter cant
            handle ''np.nan''). See documentation of pandas.Series.interpolate method for
            details on the methods associated with the different keywords.

        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
            The type of filter. Default is ‘lowpass’.

        """

        self._data[field] = butterFilter(
            self._data[field],
            cutoff=cutoff,
            nyq=nyq,
            filter_order=filter_order,
            fill_method=fill_method,
            filter_type="lowpass",
        )
        return self


def _fitPolynomial(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[int, str],
    order: int,
    min_periods: int = 0,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:

    # TODO: some (rather large) parts are functional similar to saqc.funcs.rolling.roll
    if data[field].empty:
        return data, flags

    to_fit = data[field].copy()
    regular = getFreqDelta(to_fit.index)
    if not regular:
        if isinstance(window, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                "sample series."
            )
        # get interval centers
        centers = (
            to_fit.rolling(
                pd.Timedelta(window) / 2, closed="both", min_periods=min_periods
            ).count()
        ).floor()
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        fitted = to_fit.rolling(
            pd.Timedelta(window), closed="both", min_periods=min_periods
        ).apply(polyRollerIrregular, args=(centers, order))

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = (
            centers.rolling(window, closed="both")
            .apply(center_func, raw=False)
            .astype(int)
        )
        temp = fitted.copy()
        for k in centers_iloc.iteritems():
            fitted.iloc[k[1]] = temp[k[0]]
        fitted[fitted.index[0] : fitted.index[centers_iloc[0]]] = np.nan
        fitted[fitted.index[centers_iloc[-1]] : fitted.index[-1]] = np.nan
    else:
        if isinstance(window, str):
            window = pd.Timedelta(window) // regular
        if window % 2 == 0:
            window = int(window - 1)
        if min_periods is None:
            min_periods = window
        if to_fit.shape[0] < 200000:
            numba = False
        else:
            numba = True

        val_range = np.arange(0, window)
        center_index = window // 2
        if min_periods < window:
            if min_periods > 0:
                to_fit = to_fit.rolling(
                    window, min_periods=min_periods, center=True
                ).apply(lambda x, y: x[y], raw=True, args=(center_index,))

            # we need a missing value marker that is not nan,
            # because nan values dont get passed by pandas rolling method
            miss_marker = to_fit.min()
            miss_marker = np.floor(miss_marker - 1)
            na_mask = to_fit.isna()
            to_fit[na_mask] = miss_marker
            if numba:
                fitted = to_fit.rolling(window).apply(
                    polyRollerNumba,
                    args=(miss_marker, val_range, center_index, order),
                    raw=True,
                    engine="numba",
                    engine_kwargs={"no_python": True},
                )
                # due to a tiny bug - rolling with center=True doesnt work
                # when using numba engine.
                fitted = fitted.shift(-int(center_index))
            else:
                fitted = to_fit.rolling(window, center=True).apply(
                    polyRoller,
                    args=(miss_marker, val_range, center_index, order),
                    raw=True,
                )
            fitted[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            if numba:
                fitted = to_fit.rolling(window).apply(
                    polyRollerNoMissingNumba,
                    args=(val_range, center_index, order),
                    engine="numba",
                    engine_kwargs={"no_python": True},
                    raw=True,
                )
                # due to a tiny bug - rolling with center=True doesnt work
                # when using numba engine.
                fitted = fitted.shift(-int(center_index))
            else:
                fitted = to_fit.rolling(window, center=True).apply(
                    polyRollerNoMissing,
                    args=(val_range, center_index, order),
                    raw=True,
                )

    data[field] = fitted
    worst = flags[field].rolling(window, center=True, min_periods=min_periods).max()
    flags[field] = worst

    return data, flags
