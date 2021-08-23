#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
import saqc


class Pattern(ModuleBase):
    def flagPatternByDTW(
        self, field, ref_field, max_distance=0.0, normalize=True, flag=BAD, **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagPatternByDTW", locals())

    def flagPatternByWavelet(
        self, field, ref_field, widths=(1, 2, 4, 8), waveform="mexh", flag=BAD, **kwargs
    ) -> saqc.SaQC:
        return self.defer("flagPatternByWavelet", locals())
