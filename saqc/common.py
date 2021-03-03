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
UNFLAGGED = 0.
DOUBTFUL = 25.
BAD = 255.

# aliases
GOOD = UNFLAGGED
DOUBT = DOUBTFUL


