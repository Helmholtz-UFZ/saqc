#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""The system for automated quality controll package."""

__version__ = "1.4"

# import order: from small to big
from saqc.constants import *
from saqc.core import (
    register,
    initFlagsLike,
    Flags,
    FloatTranslator,
    DmpTranslator,
    PositionalTranslator,
    SimpleTranslator,
    SaQC,
    fromConfig,
)
