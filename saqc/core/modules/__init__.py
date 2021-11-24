#! /usr/bin/env python
# -*- coding: utf-8 -*-
from saqc.core.modules.breaks import Breaks
from saqc.core.modules.noise import Noise
from saqc.core.modules.changepoints import ChangePoints
from saqc.core.modules.constants import Constants
from saqc.core.modules.curvefit import Curvefit
from saqc.core.modules.drift import Drift
from saqc.core.modules.flagtools import FlagTools
from saqc.core.modules.generic import Generic
from saqc.core.modules.interpolation import Interpolation
from saqc.core.modules.outliers import Outliers
from saqc.core.modules.pattern import Pattern
from saqc.core.modules.resampling import Resampling
from saqc.core.modules.residues import Residues
from saqc.core.modules.rolling import Rolling
from saqc.core.modules.scores import Scores
from saqc.core.modules.tools import Tools
from saqc.core.modules.transformation import Transformation
from saqc.core.register import FUNC_MAP


class FunctionsMixin(
    Breaks,
    Noise,
    ChangePoints,
    Constants,
    Curvefit,
    Drift,
    FlagTools,
    Generic,
    Interpolation,
    Outliers,
    Pattern,
    Resampling,
    Residues,
    Rolling,
    Scores,
    Tools,
    Transformation,
):
    def _defer(self, fname, flocals):
        flocals.pop("self", None)
        fkwargs = flocals.pop("kwargs", {})
        return self._wrap(FUNC_MAP[fname])(**flocals, **fkwargs)
