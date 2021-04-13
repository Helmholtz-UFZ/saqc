#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Callable

import numpy as np
import pandas as pd

from saqc.constants import *
from saqc.core.modules.base import ModuleBase


class Rolling(ModuleBase):
    def roll(
        self,
        field: str,
        winsz: Union[str, int],
        func: Callable[[pd.Series], float] = np.mean,
        eval_flags: bool = True,  # TODO: not applicable anymore
        min_periods: int = 0,
        center: bool = True,
        return_residues=False,  # TODO: this should not be public, a wrapper would be better
        flag: float = BAD,
        **kwargs
    ):
        return self.defer("roll", locals())
