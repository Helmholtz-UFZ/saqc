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
    "DOUBT",
    "ENVIRONMENT",
]


import numpy as np
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

DOUBT = DOUBTFUL  #: Alias for :py:const:`DOUBTFUL <saqc.constants.DOUBTFUL>`.

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
