#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union

from dios import DictOfSeries
from typing_extensions import Literal

from saqc.constants import BAD
import saqc


class Curvefit:
    def fitPolynomial(
        self,
        field: str,
        window: Union[int, str],
        order: int,
        set_flags: bool = True,
        min_periods: int = 0,
        return_residues: bool = False,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("fitPolynomial", locals())
