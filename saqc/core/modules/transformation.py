#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union, Tuple

import pandas as pd
from dios import DictOfSeries

from saqc.core import Flags as Flagger
from saqc.core.modules.base import ModuleBase


class Transformation(ModuleBase):

    def transform(
            self, 
            field: str,
            func: Callable[[pd.Series], pd.Series],
            partition_freq: Optional[Union[float, str]] = None,
            **kwargs
    ) -> Tuple[DictOfSeries, Flagger]:
        return self.defer("transform", locals())
