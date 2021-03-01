#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

import pandas as pd

from saqc.core.modules.base import ModuleBase


class Transformation(ModuleBase):

    def transform(
            self,
            field: str,
            func: Callable[[pd.Series], pd.Series],
            partition_freq: Optional[Union[float, str]] = None,
            **kwargs
    ):
        return self.defer("transform", locals())