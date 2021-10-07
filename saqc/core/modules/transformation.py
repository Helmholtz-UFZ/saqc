#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Union

import pandas as pd

from saqc.core.modules.base import ModuleBase
import saqc


class Transformation(ModuleBase):
    def transform(
        self,
        field: str,
        func: Callable[[pd.Series], pd.Series],
        freq: Optional[Union[float, str]] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self.defer("transform", locals())
