#!/usr/bin/env python

from typing import Dict, Optional
from typing_extensions import Literal

from saqc.core.lib import SaQCFunction

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, SaQCFunction] = {}


def register(masking: Literal["all", "field", "none"]="all", module: Optional[str]=None):

    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"
        FUNC_MAP[func_name] = SaQCFunction(func_name, masking, func)
        return func

    return inner
