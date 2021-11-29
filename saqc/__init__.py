#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""The system for automated quality controll package."""

from saqc.version import __version__

from saqc.constants import (
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)

# import order: from small to big, to a void cycles
from saqc.core import (
    Flags,
    SaQC,
    fromConfig,
)
