#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""The system for automated quality controll package."""

__version__ = "1.4"

# import order: from small to big
from saqc.constants import (
    UNTOUCHED,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.core import (
    register,
    flagging,
    processing,
    Flags,
    SaQC,
    fromConfig,
)
