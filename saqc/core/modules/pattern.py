#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase


class Pattern(ModuleBase):

    def flagPatternByDTW(
            self, 
            field: str,
            ref_field: str,
            widths: Sequence[int] = (1, 2, 4, 8),
            waveform: str = "mexh",
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flagPatternByDTW", locals())

    def flagPatternByWavelet(
            self, 
            field: str,
            ref_field: str,
            max_distance: float = 0.03,
            normalize: bool = True,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flagPatternByWavelet", locals())
