#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
"""The system for automated quality controll package."""

from saqc.version import __version__

from saqc.constants import UNFLAGGED, GOOD, DOUBTFUL, BAD, FILTER_NONE, FILTER_ALL

# import order: from small to big, to a void cycles
from saqc.core import (
    Flags,
    SaQC,
    fromConfig,
)
