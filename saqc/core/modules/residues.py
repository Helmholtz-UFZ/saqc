#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Callable
from typing_extensions import Literal

import numpy as np

from saqc.core.modules.base import ModuleBase


class Residues(ModuleBase):

    def calculatePolynomialResidues(
            self,
            field: str,
            winsz: Union[str, int],
            polydeg: int,
            numba: Literal[True, False, "auto"] = "auto",
            eval_flags: bool = True,
            min_periods: Optional[int] = 0,
            **kwargs
    ):
        return self.defer("calculatePolynomialResidues", locals())

    def calculateRollingResidues(
            self,
            field: str,
            winsz: Union[str, int],
            func: Callable[[np.array], np.array] = np.mean,
            eval_flags: bool = True,
            min_periods: Optional[int] = 0,
            center: bool = True,
            **kwargs
    ):
        return self.defer("calculateRollingResidues", locals())
