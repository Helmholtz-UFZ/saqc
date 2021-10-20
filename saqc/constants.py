#!/usr/bin/env python

__all__ = [
    "UNTOUCHED",
    "UNFLAGGED",
    "DOUBTFUL",
    "BAD",
    "GOOD",
    "DOUBT",
]

import numpy as np

UNTOUCHED = np.nan
UNFLAGGED = -np.inf
GOOD = 0
DOUBTFUL = 25.0
BAD = 255.0

# aliases
DOUBT = DOUBTFUL
