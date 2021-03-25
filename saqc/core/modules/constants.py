#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

from dios import DictOfSeries

from saqc.constants import *
from saqc.core import Flags as Flagger
from saqc.core.modules.base import ModuleBase
from saqc.lib.types import FreqString, ColumnName


class Constants(ModuleBase):

    def flagByVariance(
            self, 
            field: ColumnName,
            window: FreqString = "12h",
            thresh: float = 0.0005,
            max_missing: int = None,
            max_consec_missing: int = None,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagByVariance", locals())

    def flagConstants(
            self, 
            field: ColumnName,
            thresh: float,
            window: FreqString,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("flagConstants", locals())
