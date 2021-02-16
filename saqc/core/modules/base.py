#! /usr/bin/env python
# -*- coding: utf-8 -*-

from saqc.core.register import FUNC_MAP


class ModuleBase:

    def __init__(self, obj):
        self.obj = obj

    def __str__(self):
        return self.__class__.__name__.lower()

    def __getattr__(self, key):
        raise AttributeError(f"'SaQC.{self}' object has no attribute '{key}'")

    def defer(self, fname, flocals):
        flocals.pop("self", None)
        fkwargs = flocals.pop("kwargs", {})
        return self.obj._wrap(FUNC_MAP[f"{self}.{fname}"])(**flocals, **fkwargs)
