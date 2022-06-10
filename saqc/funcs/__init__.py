#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

# imports needed to make the functions register themself
from saqc.core.register import register
from saqc.funcs.breaks import *
from saqc.funcs.changepoints import *
from saqc.funcs.constants import *
from saqc.funcs.curvefit import *
from saqc.funcs.drift import *
from saqc.funcs.flagtools import *
from saqc.funcs.generic import *
from saqc.funcs.interpolation import *
from saqc.funcs.noise import *
from saqc.funcs.outliers import *
from saqc.funcs.pattern import *
from saqc.funcs.resampling import *
from saqc.funcs.residuals import *
from saqc.funcs.rolling import *
from saqc.funcs.scores import *
from saqc.funcs.tools import *
from saqc.funcs.transformation import *
