#!/usr/bin/env python

from typing import Dict, Any

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Any] = {}


def register(masking='all'):
    def inner(func):
        ctrl_kws = dict(masking=masking)
        FUNC_MAP[func.__name__] = dict(func=func, ctrl_kws=ctrl_kws)
        return func

    return inner
