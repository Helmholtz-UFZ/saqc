#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
from typing_extensions import Literal

import saqc
from saqc.lib.types import FreqString
from saqc.core.modules.base import ModuleBase


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
        mode: Literal["periodic", "mask_field"],
        mask_field: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        closed: bool = True,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("mask", locals())

    def plot(
        self,
        field: str,
        path: Optional[str] = None,
        max_gap: Optional[FreqString] = None,
        stats: bool = False,
        history: Optional[Literal["valid", "complete"]] = "valid",
        ax_kwargs: Optional[dict] = None,
        fig_kwargs: Optional[dict] = None,
        scatter_kwargs: Optional[dict] = None,
        stats_dict: Optional[dict] = None,
        store_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> saqc.SaQC:
        return self.defer("plot", locals())
