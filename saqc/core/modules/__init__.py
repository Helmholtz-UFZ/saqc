#! /usr/bin/env python
# -*- coding: utf-8 -*-

from saqc.core.modules.breaks import Breaks
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


class FuncModules:
    def __init__(self, obj):
        # our testing modules
        self.breaks = Breaks(obj)
        self.changepoints = ChangePoints(obj)
        self.constants = Constants(obj)
        self.curvefit = Curvefit(obj)
        self.drift = Drift(obj)
        self.flagtools = FlagTools(obj)
        self.generic = Generic(obj)
        self.interpolation = Interpolation(obj)
        self.outliers = Outliers(obj)
        self.pattern = Pattern(obj)
        self.resampling = Resampling(obj)
        self.residues = Residues(obj)
        self.rolling = Rolling(obj)
        self.scores = Scores(obj)
        self.tools = Tools(obj)
        self.transformation = Transformation(obj)
