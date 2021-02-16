#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from saqc.core.modules.base import ModuleBase


class Breaks(ModuleBase):

    def flagMissing(self, field: str, nodata=np.nan, **kwargs):
        return self.defer("flagMissing", locals())

    def flagIsolated(self, field: str, gap_window: str, group_window: str, **kwargs):
        return self.defer("flagIsolated", locals())

    def flagJumps(self, field: str, thresh: float, winsz: str, min_periods: int = 1, **kwargs):
        return self.defer("flagJumps", locals())
