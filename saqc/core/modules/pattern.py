#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence

from saqc.core.modules.base import ModuleBase


class Pattern(ModuleBase):

    def flagPatternByDTW(
            self,
            field: str,
            ref_field: str,
            widths: Sequence[int] = (1, 2, 4, 8),
            waveform: str = "mexh",
            **kwargs
    ):
        return self.defer("flagPatternByDTW", locals())

    def flagPatternByWavelet(
            self,
            field: str,
            ref_field: str,
            max_distance: float = 0.03,
            normalize: bool = True,
            **kwargs
    ):
        return self.defer("flagPatternByWavelet", locals())
