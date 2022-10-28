#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
The module comprises flag value constants in use throughout saqc.

Flagging Constants
------------------
* :py:const:`~saqc.constants.UNFLAGGED`: indicates that no flag has been assigned yet
* :py:const:`~saqc.constants.GOOD`: the lowest possible flag value larget than :py:const:`~saqc.constants.UNFLAGGED`
* :py:const:`~saqc.constants.DOUBFUL`: a modest flag value, usually indicating some sort of suspiciousness
* :py:const:`~saqc.constants.BAD`: the highest flag value available

The flagging constants are ordered (from "worse" to "best") as:

:py:const:`~saqc.constants.BAD` > :py:const:`~saqc.constants.DOUBTFUL` > :py:const:`~saqc.constants.GOOD` > :py:const:`~saqc.constants.UNFLAGGED`

Filtering Constants
-------------------
* :py:const:`~saqc.constants.FILTER_ALL`: mask/filter all flagged data
* :py:const:`~saqc.constants.FILTER_NONE`: mask/filter no data
"""

__all__ = [
    "UNFLAGGED",
    "DOUBTFUL",
    "BAD",
    "GOOD",
    "ENVIRONMENT",
    "FILTER_ALL",
    "FILTER_NONE",
]


import numpy as np
import scipy.stats as st

import saqc.lib.ts_operators as ts_ops

# ----------------------------------------------------------------------
# global flag constants
# ----------------------------------------------------------------------

UNFLAGGED = -np.inf
GOOD = 0
DOUBTFUL = 25.0
BAD = 255.0

# ----------------------------------------------------------------------
# global dfilter constants
# ----------------------------------------------------------------------

FILTER_ALL = -np.inf
FILTER_NONE = np.inf

# ----------------------------------------------------------------------
# other
# ----------------------------------------------------------------------
def clip(series, lower=None, upper=None):
    return series.clip(lower=lower, upper=upper)


ENVIRONMENT = {
    # Infinity constant
    "inf": np.inf,
    "INF": np.inf,
    # Not A number Constant.
    "NAN": np.nan,
    "nan": np.nan,
    # Pointwise absolute Value Function.
    "abs": np.abs,
    # Maximum Value Function. Ignores NaN.
    "max": np.nanmax,
    # Minimum Value Function. Ignores NaN.
    "min": np.nanmin,
    # Mean Value Function. Ignores NaN.
    "mean": np.nanmean,
    # Summation. Ignores NaN.
    "sum": np.nansum,
    # Standart Deviation. Ignores NaN.
    "len": len,
    # Pointwise Exponential.
    "exp": np.exp,
    # Pointwise Logarithm.
    "log": np.log,
    # Logarithm, returning NaN for zero input, instead of -inf.
    "nanLog": ts_ops.zeroLog,
    # Standart Deviation. Ignores NaN.
    "std": np.nanstd,
    # Variance. Ignores NaN.
    "var": np.nanvar,
    # Median. Ignores NaN.
    "median": np.nanmedian,
    # Count Number of values. Ignores NaNs.
    "count": ts_ops.count,
    # Identity.
    "id": ts_ops.identity,
    # Returns a Series` diff.
    "diff": ts_ops.difference,
    # Scales data to [0,1] Interval.
    "scale": ts_ops.normScale,
    # Standardize with Standart Deviation.
    "zScore": lambda x: st.zscore(x, nan_policy="omit"),
    # Standardize with Median and MAD.
    "madScore": ts_ops.standardizeByMedian,
    # Standardize with Median and inter quantile range.
    "iqsScore": ts_ops.standardizeByIQR,
    "clip": clip,
    "GOOD": GOOD,
    "BAD": BAD,
    "UNFLAGGED": UNFLAGGED,
    "DOUBTFUL": DOUBTFUL,
    "FILTER_ALL": FILTER_ALL,
    "FILTER_NONE": FILTER_NONE,
}
