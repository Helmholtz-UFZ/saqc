#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from saqc.funcs.breaks import BreaksMixin
from saqc.funcs.changepoints import ChangepointsMixin
from saqc.funcs.constants import ConstantsMixin
from saqc.funcs.curvefit import CurvefitMixin
from saqc.funcs.drift import DriftMixin
from saqc.funcs.flagtools import FlagtoolsMixin
from saqc.funcs.generic import GenericMixin
from saqc.funcs.interpolation import InterpolationMixin
from saqc.funcs.noise import NoiseMixin
from saqc.funcs.outliers import OutliersMixin
from saqc.funcs.pattern import PatternMixin
from saqc.funcs.resampling import ResamplingMixin
from saqc.funcs.residuals import ResidualsMixin
from saqc.funcs.rolling import RollingMixin
from saqc.funcs.scores import ScoresMixin
from saqc.funcs.tools import ToolsMixin
from saqc.funcs.transformation import TransformationMixin


class FunctionsMixin(
    BreaksMixin,
    ChangepointsMixin,
    ConstantsMixin,
    CurvefitMixin,
    DriftMixin,
    FlagtoolsMixin,
    GenericMixin,
    InterpolationMixin,
    NoiseMixin,
    OutliersMixin,
    PatternMixin,
    ResamplingMixin,
    ResidualsMixin,
    RollingMixin,
    ScoresMixin,
    ToolsMixin,
    TransformationMixin,
):
    pass
