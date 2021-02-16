#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union

from typing_extensions import Literal

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
            **kwargs):
        return self.defer("fitPolynomial", locals())
