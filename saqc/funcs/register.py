#!/usr/bin/env python

from functools import partial
from inspect import signature, _VAR_KEYWORD


class Partial(partial):
    def __init__(self, func, *args, **kwargs):
        self._signature = signature(func)

    @property
    def signature(self):
        out = []
        for k, v in self._signature.parameters.items():
            if v.kind != _VAR_KEYWORD:
                out.append(k)
        return tuple(out)


# NOTE: will be filled by calls to register
FUNC_MAP = {}


def register():
    def outer(func):
        name = func.__name__
        func = Partial(func, func_name=name)
        FUNC_MAP[name] = func

        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return outer
