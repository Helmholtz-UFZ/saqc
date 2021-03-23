#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Tuple

from dios import DictOfSeries

from saqc import Flagger
from saqc.core.modules.base import ModuleBase


class Pattern(ModuleBase):

    def flagPatternByDTW(
            self, 
            field: str,
            ref_field: str,
            widths: Sequence[int]=(1, 2, 4, 8),
            waveform: str="mexh",
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagPatternByDTW", locals())

    def flagPatternByWavelet(
            self, 
            field: str,
            ref_field: str,
            max_distance: float=0.03,
            normalize: bool=True,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagPatternByWavelet", locals())
