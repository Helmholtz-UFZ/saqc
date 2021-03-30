#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable

import numpy as np
from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase


class Residues(ModuleBase):

    def calculatePolynomialResidues(
            self, 
            field: str,
            winsz: Union[str, int],
            polydeg: int,
            numba: Literal[True, False, "auto"] = "auto",  # TODO: rm, not a a user decision
            eval_flags: bool = True,  # TODO, not valid anymore, if still needed, maybe assign user-passed ``flag``?
            min_periods: Optional[int] = 0,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("calculatePolynomialResidues", locals())

    def calculateRollingResidues(
            self, 
            field: str,
            winsz: Union[str, int],
            func: Callable[[np.ndarray], np.ndarray] = np.mean,
            eval_flags: bool = True,
            min_periods: Optional[int] = 0,
            center: bool = True,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("calculateRollingResidues", locals())
