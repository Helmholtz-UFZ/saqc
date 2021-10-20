#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional, Union

import pandas as pd

import saqc


class Transformation:
    def transform(
        self,
        field: str,
        func: Callable[[pd.Series], pd.Series],
        freq: Optional[Union[float, str]] = None,
        **kwargs
    ) -> saqc.SaQC:
        return self._defer("transform", locals())
