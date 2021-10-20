#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from saqc.constants import BAD, UNFLAGGED
import saqc
from saqc.lib.types import FreqString


class Breaks:
    def flagMissing(
        self, field: str, flag: float = BAD, to_mask: float = UNFLAGGED, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagMissing", locals())

    def flagIsolated(
        self,
        field: str,
        gap_window: FreqString,
        group_window: FreqString,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagIsolated", locals())

    def flagJumps(
        self,
        field: str,
        thresh: float,
        window: FreqString,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagJumps", locals())
