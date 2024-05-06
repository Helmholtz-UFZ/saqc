#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

# isort: skip_file

"""The System for automated Quality Control package."""

__all__ = [
    "BAD",
    "DOUBTFUL",
    "GOOD",
    "UNFLAGGED",
    "FILTER_ALL",
    "FILTER_NONE",
    "Flags",
    "DictOfSeries",
    "SaQC",
    "DmpScheme",
    "FloatScheme",
    "PositionalScheme",
    "SimpleScheme",
    "AnnotatedFloatScheme",
    "fromConfig",
]

from saqc.constants import BAD, DOUBTFUL, FILTER_ALL, FILTER_NONE, GOOD, UNFLAGGED
from saqc.core import Flags, DictOfSeries, SaQC
from saqc.core.translation import (
    DmpScheme,
    FloatScheme,
    PositionalScheme,
    SimpleScheme,
    AnnotatedFloatScheme,
)
from saqc.parsing.reader import fromConfig
from . import _version

__version__ = _version.get_versions()["version"]
