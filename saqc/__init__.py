#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

# isort: skip_file

"""The System for automated Quality Control package."""

from saqc.constants import BAD, DOUBTFUL, FILTER_ALL, FILTER_NONE, GOOD, UNFLAGGED
from saqc.exceptions import ParsingError
from saqc.core import Flags, DictOfSeries, SaQC
from saqc.core.translation import DmpScheme, FloatScheme, PositionalScheme, SimpleScheme
from saqc.parsing.reader import fromConfig
from saqc.version import __version__

from . import _version

__version__ = _version.get_versions()["version"]
