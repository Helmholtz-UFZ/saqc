#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

"""The System for automated Quality Control package."""

from saqc.constants import BAD, DOUBTFUL, FILTER_ALL, FILTER_NONE, GOOD, UNFLAGGED

# import order: from small to big, to a void cycles
from saqc.core import Flags, SaQC, fromConfig
from saqc.version import __version__
