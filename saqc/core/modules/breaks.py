#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from saqc.constants import BAD, FILTER_ALL
import saqc


class Breaks:
    def flagMissing(
        self, field: str, flag: float = BAD, dfilter: float = FILTER_ALL, **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagMissing", locals())

    def flagIsolated(
        self,
        field: str,
        gap_window: str,
        group_window: str,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagIsolated", locals())

    def flagJumps(
        self,
        field: str,
        thresh: float,
        window: str,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("flagJumps", locals())
