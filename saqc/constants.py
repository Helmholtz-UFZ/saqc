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
    "FILTER_ALL",
    "FILTER_NONE",
]


import numpy as np

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
