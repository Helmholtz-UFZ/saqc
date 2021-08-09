#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple, Literal

from dios import DictOfSeries

from saqc.lib.types import FreqString
from saqc.core import Flags
from saqc.core.modules.base import ModuleBase
import saqc


class Tools(ModuleBase):
    def copy(self, field: str, new_field: str, **kwargs) -> saqc.SaQC:
        return self.defer("copy", locals())

    def drop(self, field: str, **kwargs) -> saqc.SaQC:
        return self.defer("drop", locals())

    def rename(self, field: str, new_name: str, **kwargs) -> saqc.SaQC:
        return self.defer("rename", locals())

    def mask(
        self,
        field: str,
        mode: Literal["periodic", "mask_var"],
        mask_var: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        include_bounds: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("mask", locals())

    def plot(
        self,
        field: str,
        save_path: Optional[str] = None,
        max_gap: Optional[FreqString] = None,
        stats: bool = False,
        plot_kwargs: Optional[dict] = None,
        fig_kwargs: Optional[dict] = None,
        stats_dict: Optional[dict] = None,
        save_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("plot", locals())
