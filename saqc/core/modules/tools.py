#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
from typing_extensions import Literal

import saqc
from saqc.lib.types import FreqString


class Tools:
    def copyField(self, field: str, new_field: str, **kwargs) -> saqc.SaQC:
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
        max_gap: Optional[FreqString] = None,
        stats: bool = False,
        plot_kwargs: Optional[dict] = None,
        fig_kwargs: Optional[dict] = None,
        stats_dict: Optional[dict] = None,
        store_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("plot", locals())
