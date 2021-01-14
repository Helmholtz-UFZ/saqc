#!/usr/bin/env python

from typing import Dict, Any

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Any] = {}


def register(masking='all'):
    def inner(func):
        FUNC_MAP[func.__name__] = {"func": func, "masking": masking}
        return func
    return inner

