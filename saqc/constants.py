#!/usr/bin/env python

__all__ = [
    "UNTOUCHED",
    "UNFLAGGED",
    "DOUBTFUL",
    "BAD",
    "GOOD",
    "DOUBT",
    "ENVIRONMENT",
]


import numpy as np
import saqc.lib.ts_operators as ts_ops


UNTOUCHED = np.nan
UNFLAGGED = -np.inf
GOOD = 0
DOUBTFUL = 25.0
BAD = 255.0

# aliases
DOUBT = DOUBTFUL

ENVIRONMENT = {
    "NAN": np.nan,
    "abs": np.abs,
    "max": np.nanmax,
    "min": np.nanmin,
    "mean": np.nanmean,
    "sum": np.nansum,
    "std": np.nanstd,
    "len": len,
    "exp": np.exp,
    "log": np.log,
    "var": np.nanvar,
    "median": np.nanmedian,
    "first": ts_ops.first,
    "last": ts_ops.last,
    "count": ts_ops.count,
    "deltaT": ts_ops.deltaT,
    "id": ts_ops.identity,
    "diff": ts_ops.difference,
    "relDiff": ts_ops.relativeDifference,
    "deriv": ts_ops.derivative,
    "rateOfChange": ts_ops.rateOfChange,
    "scale": ts_ops.scale,
    "normScale": ts_ops.normScale,
    "meanStandardize": ts_ops.standardizeByMean,
    "medianStandardize": ts_ops.standardizeByMedian,
    "zLog": ts_ops.zeroLog,
    "GOOD": GOOD,
    "BAD": BAD,
    "UNFLAGGED": UNFLAGGED,
    "DOUBTFUL": DOUBTFUL,
}
