#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from dios import DictOfSeries

from saqc.constants import *
from saqc.core.modules.base import ModuleBase
from saqc.core import Flags
from saqc.lib.types import FreqString, IntegerWindow, ColumnName


class Breaks(ModuleBase):

    def flagMissing(
            self,
            field: ColumnName,
            nodata: float = np.nan,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagMissing", locals())

    def flagIsolated(
            self,
            field: ColumnName,
            gap_window: FreqString,
            group_window: FreqString,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagIsolated", locals())

    def flagJumps(
            self,
            field: ColumnName,
            thresh: float,
            winsz: FreqString,
            min_periods: IntegerWindow=1,
            flag: float = BAD,
            **kwargs
    ) -> Tuple[DictOfSeries, Flags]:
        return self.defer("flagJumps", locals())
