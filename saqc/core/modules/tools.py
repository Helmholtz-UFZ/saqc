#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
from typing_extensions import Literal

import saqc
import numpy as np


class Tools:
    def copyField(self, field: str, target: str, **kwargs) -> saqc.SaQC:
        return self._defer("copyField", locals())

    def dropField(self, field: str, **kwargs) -> saqc.SaQC:
        return self._defer("dropField", locals())

    def renameField(self, field: str, new_name: str, **kwargs) -> saqc.SaQC:
        return self._defer("renameField", locals())

    def maskTime(
        self,
        field: str,
        mode: Literal["periodic", "mask_field"],
        mask_field: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        closed: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("maskTime", locals())

    def plot(
        self,
        field: str,
        path: Optional[str] = None,
        max_gap: Optional[str] = None,
        stats: bool = False,
        history: Optional[Literal["valid", "complete", "clear"]] = "valid",
        xscope: Optional[slice] = None,
        phaseplot: Optional[str] = None,
        stats_dict: Optional[dict] = None,
        store_kwargs: Optional[dict] = None,
        to_mask: Optional[float] = np.inf,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("plot", locals())
