#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from saqc.constants import UNFLAGGED, BAD
from saqc.core.modules.base import ModuleBase
import saqc


class Generic(ModuleBase):
    def process(
        self,
        field: str,
        func: Callable[[pd.Series], pd.Series],
        to_mask: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("process", locals())

    def flag(
        self,
        field: str,
        func: Callable[[pd.Series], pd.Series],
        flag: float = BAD,
        to_mask: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("flag", locals())
