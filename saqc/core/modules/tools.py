#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
from typing_extensions import Literal

from saqc.core.modules.base import ModuleBase


class Tools(ModuleBase):
    def copy(
            self,
            field: str,
            new_field: str,
            **kwargs
    ):
        return self.defer("copy", locals())

    def drop(
            self,
            field: str,
            **kwargs
    ):
        return self.defer("drop", locals())

    def rename(
            self,
            field: str,
            new_name: str,
            **kwargs
    ):
        return self.defer("rename", locals())

    def mask(
            self,
            field: str,
            mode: Literal["periodic", "mask_var"],
            mask_var: Optional[str] = None,
            period_start: Optional[str] = None,
            period_end: Optional[str] = None,
            include_bounds: bool = True
    ):
        return self.defer("mask", locals())