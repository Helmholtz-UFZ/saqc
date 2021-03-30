#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from saqc.constants import BAD
from saqc.core.modules.base import ModuleBase
from saqc.lib.types import FreqString, IntegerWindow, ColumnName


class Breaks(ModuleBase):

    def flagMissing(
            self, 
            field: ColumnName,
            nodata: float = np.nan,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flagMissing", locals())

    def flagIsolated(
            self, 
            field: ColumnName,
            gap_window: FreqString,
            group_window: FreqString,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flagIsolated", locals())

    def flagJumps(
            self, 
            field: ColumnName,
            thresh: float,
            winsz: FreqString,
            min_periods: IntegerWindow = 1,
            flag: float = BAD,
            **kwargs
    ) -> SaQC:
        return self.defer("flagJumps", locals())
