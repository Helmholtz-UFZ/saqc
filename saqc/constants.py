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
]

import numpy as np

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
