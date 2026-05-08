#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from saqc.core import DictOfSeries, Flags, register
from saqc.lib.tools import getFreqDelta
from saqc.lib.ts_operators import (
    butterFilter,
    polyRoller,
    polyRollerIrregular,
    polyRollerNoMissing,
)
from saqc.lib.types import (
    FILL_METHODS,
    Float,
    FreqStr,
    Int,
    OffsetStr,
    SaQCFields,
    ValidatePublicMembers,
)

DEFAULT_MOMENT = dict(
    pretrained_model_name_or_path="AutonLab/MOMENT-1-large", revision="main"
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saqc import SaQC


class CurvefitMixin(ValidatePublicMembers):
    @register(mask=["field"], demask=[], squeeze=[])
    def fitPolynomial(
        self: SaQC,
        field: SaQCFields,
        window: OffsetStr | (Int >= 0),
        order: Int >= 1,
        min_periods: Int >= 0 = 0,
        **kwargs,
    ) -> SaQC:
        """
        Fit a polynomial model to the data.

        The fit is calculated by fitting a polynomial of degree `order` to a data slice
        of extension `window`, centered around each timestamp.

        For regularly sampled data:

        * If missing values are rare or residuals for windows with missing values are
          not needed, performance can be increased by setting min_periods=window.
        * The initial and final ``window``//2 timestamps do not get fitted.
        * Each residual is assigned the worst flag present in the corresponding interval
          of the original data.

        Parameters
        ----------
        window :
            Extension of the fitting window.

            If an integer is passed, it represents the number
            of timestamps in each window. If an offset string is passed, it represents the
            window's temporal extent. The window is centered around the timestamp being fitted.
            For uniformly sampled data, an odd number of timestamps is always used to constitute a window (subtracted by 1,
            if the total is even).

        order :
            Degree of the fitted polynomial.

        min_periods :
            Minimum population for fitting windows.

            Windows with fewer timestamps will result in NaN valued smoothing points. Passing 0 disables this
            check and may result in overfitting for sparse windows.
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
        self: SaQC,
        field: SaQCFields,
        cutoff: (Float >= 0) | FreqStr,
        nyq: Float >= 0 = 0.5,
        filter_order: Int >= 1 = 2,
        fill_method: FILL_METHODS = "linear",
        **kwargs,
    ) -> SaQC:
        """
        Filter and smooth data with Butterworth filter.

        Derive a smoothed version of the data by cutting off frequencies of its spectral representation that
        exceed a cutoff frequency.

        Parameters
        ----------
        cutoff :
            The cutoff-frequency.

            Has to be either an offset freq string, or be expressed in multiples of the sampling rate.

        nyq :
            The niquist-frequency.

            expressed in multiples if the sampling rate.

        fill_method :
            Fill method applied pre-filtering.

            Since butterworth filtering cant handle `np.nan` values or irregularly sampled data, an imputation method
            for gaps should be assigned here. See documentation of pandas.Series.interpolate method for
            details on the methods associated with the different keywords.

        Note
        ----
        The data is expected to be regularly sampled.
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
    window: int | str,
    order: int,
    min_periods: int = 0,
    **kwargs,
) -> tuple[DictOfSeries, Flags]:
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
        centers = to_fit.rolling(
            pd.Timedelta(window) / 2, closed="both", min_periods=min_periods
        ).count()
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        fitted = to_fit.rolling(
            pd.Timedelta(window), closed="both", min_periods=min_periods, center=True
        ).apply(polyRollerIrregular, args=(centers, order))
    else:  # if regular
        if isinstance(window, str):
            window = pd.Timedelta(window) // regular
        if window % 2 == 0:
            window = int(window - 1)
        if min_periods is None:
            min_periods = window

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

            fitted = to_fit.rolling(window, center=True).apply(
                polyRoller,
                args=(miss_marker, val_range, center_index, order),
                raw=True,
            )
            fitted[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            fitted = to_fit.rolling(window, center=True).apply(
                polyRollerNoMissing,
                args=(val_range, center_index, order),
                raw=True,
            )

    data[field] = fitted
    worst = flags[field].rolling(window, center=True, min_periods=min_periods).max()
    flags[field] = worst

    return data, flags
