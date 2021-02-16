#! /usr/bin/env python
# -*- coding: utf-8 -*-

from saqc.core.modules.base import ModuleBase


class Constants(ModuleBase):

    def flagByVariance(
            self, field: str,
            window: str = "12h",
            thresh: float = 0.0005,
            max_missing: int = None,
            max_consec_missing: int = None,
            **kwargs
    ):
        return self.defer("flagByVariance", locals())

    def flagConstants(self, field: str, thresh: float, window: str, **kwargs):
        return self.defer("flagConstants", locals())
