#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union

from typing_extensions import Literal

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase


class Curvefit(ModuleBase):
    def fitPolynomial(
        self,
        field: str,
        winsz: Union[int, str],
        polydeg: int,
        numba: Literal[True, False, "auto"] = "auto",
        eval_flags: bool = True,
        min_periods: int = 0,
        return_residues: bool = False,
        flag: float = BAD,
        **kwargs
    ) -> SaQC:
        return self.defer("fitPolynomial", locals())
