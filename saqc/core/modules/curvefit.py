#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union, Literal

from dios import DictOfSeries
from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc


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
    ) -> saqc.SaQC:
        return self.defer("fitPolynomial", locals())
