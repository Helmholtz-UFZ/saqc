#! /usr/bin/env python
# -*- coding: utf-8 -*-

from saqc.core.register import flagging, processing
from saqc.core.flags import Flags, initFlagsLike
from saqc.core.core import SaQC
from saqc.core.translator import (
    FloatTranslator,
    DmpTranslator,
    PositionalTranslator,
    SimpleTranslator,
)
from saqc.core.reader import fromConfig
