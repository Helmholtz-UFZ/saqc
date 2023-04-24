#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st

import saqc.lib.ts_operators as ts_ops
from saqc import BAD, DOUBTFUL, FILTER_ALL, FILTER_NONE, GOOD, UNFLAGGED


def clip(series, lower=None, upper=None):
    return series.clip(lower=lower, upper=upper)


def zscore(obj):
    return st.zscore(obj, nan_policy="omit")


def cv(series: pd.Series) -> pd.Series:
    """
    calculates the coefficient of variation on a min-max scaled time series
    """
    series_ = (series - series.min()) / (series.max() - series.min())
    return series_.std() / series_.mean()


ENVIRONMENT = {
    # Infinity constant
    "inf": np.inf,
    "INF": np.inf,
    # Not a number constant.
    "NAN": np.nan,
    "nan": np.nan,
    # Absolute value function.
    "abs": np.abs,
    # Maximum value function. Ignores NaN.
    "max": np.nanmax,
    # Minimum Value function. Ignores NaN.
    "min": np.nanmin,
    # Mean value function. Ignores NaN.
    "mean": np.nanmean,
    # Summation. Ignores NaN.
    "sum": np.nansum,
    # Standard deviation. Ignores NaN.
    "len": len,
    # exponential function.
    "exp": np.exp,
    # Logarithm.
    "log": np.log,
    # Logarithm, returning NaN for zero input, instead of -inf.
    "nanLog": ts_ops.zeroLog,
    # Standard deviation. Ignores NaN.
    "std": np.nanstd,
    # Variance. Ignores NaN.
    "var": np.nanvar,
    # Coefficient of variation.
    "cv": cv,
    # Median. Ignores NaN.
    "median": np.nanmedian,
    # Count Number of values. Ignores NaNs.
    "count": ts_ops.count,
    # Identity.
    "id": ts_ops.identity,
    # Returns a series` diff.
    "diff": ts_ops.difference,
    # Scales data to [0,1] interval.
    "scale": ts_ops.normScale,
    # Standardize with standard deviation.
    "zScore": zscore,
    # Standardize with median and MAD.
    "madScore": ts_ops.standardizeByMedian,
    # Standardize with median and inter quantile range.
    "iqsScore": ts_ops.standardizeByIQR,
    "clip": clip,
    "GOOD": GOOD,
    "BAD": BAD,
    "UNFLAGGED": UNFLAGGED,
    "DOUBTFUL": DOUBTFUL,
    "FILTER_ALL": FILTER_ALL,
    "FILTER_NONE": FILTER_NONE,
}
