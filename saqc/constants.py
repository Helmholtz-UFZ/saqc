#!/usr/bin/env python
"""
The module comprises flag value constants in use throughout saqc.
The constants order as follows (from "worse" to "best"):

:py:const:`~saqc.constants.BAD` > :py:const:`~saqc.constants.DOUBTFUL` > :py:const:`~saqc.constants.GOOD` >
:py:const:`~saqc.constants.UNFLAGGED` (> :py:const:`~saqc.constants.UNTOUCHED`)
"""

__all__ = [
    "UNTOUCHED",
    "UNFLAGGED",
    "DOUBTFUL",
    "BAD",
    "GOOD",
    "ENVIRONMENT",
]


import numpy as np
import scipy.stats as st
import saqc.lib.ts_operators as ts_ops


#: Internal :py:mod:`flag level constant <saqc.constants>`.
#: When returned by a test, it indicates, that the test did not consider to flag the respective value
UNTOUCHED = np.nan

#: A :py:mod:`flag level constant <saqc.constants>`
#: , evaluating to the level, that indicates, no flag has been assigned to yet.
UNFLAGGED = -np.inf

#: A :py:mod:`flag level constant <saqc.constants>`
#: , evaluating to the lowest level level of flagging, that is
#: not :py:const:`UNFLAGGED <saqc.constants.UNFLAGGED>`.
GOOD = 0

#: A :py:mod:`flag level constant <saqc.constants>`
#: , evaluating to a somewhat modest flag level.
DOUBTFUL = 25.0

#: A :py:mod:`flag level constant <saqc.constants>`
#: , evaluating to the highest (internal) flag level available.
BAD = 255.0


ENVIRONMENT = {
    # Not A number Constant.
    "NAN": np.nan,
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
    "GOOD": GOOD,
    "BAD": BAD,
    "UNFLAGGED": UNFLAGGED,
    "DOUBTFUL": DOUBTFUL,
}
