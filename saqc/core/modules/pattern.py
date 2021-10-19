#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

from saqc.constants import BAD
import saqc


class Pattern:
    def flagPatternByDTW(
        self,
        field,
        reference,
        max_distance=0.0,
        normalize=True,
        plot=False,
        flag=BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagPatternByDTW", locals())

    def flagPatternByWavelet(
        self, field, reference, widths=(1, 2, 4, 8), waveform="mexh", flag=BAD, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagPatternByWavelet", locals())
