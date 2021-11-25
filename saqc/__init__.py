#! /usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.4"

# import order: from small to big
from saqc.constants import (
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)

from saqc.core import (
    Flags,
    SaQC,
    fromConfig,
)
