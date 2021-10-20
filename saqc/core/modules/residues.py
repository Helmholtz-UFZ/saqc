#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Union, Callable

import numpy as np
from typing_extensions import Literal

from saqc.constants import BAD
import saqc


class Residues:
    def calculatePolynomialResidues(
        self,
        field: str,
        window: Union[str, int],
        order: int,
        set_flags: bool = True,  # TODO, not valid anymore, if still needed, maybe assign user-passed ``flag``?
        min_periods: Optional[int] = 0,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("calculatePolynomialResidues", locals())

    def calculateRollingResidues(
        self,
        field: str,
        window: Union[str, int],
        func: Callable[[np.ndarray], np.ndarray] = np.mean,
        set_flags: bool = True,
        min_periods: Optional[int] = 0,
        center: bool = True,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("calculateRollingResidues", locals())
