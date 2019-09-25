#!/usr/bin/env python

"""
docstring: TODO
"""

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"

from functools import partial

# NOTE: will be filled by calls to register
FUNC_MAP = {}


def register(name):

    def outer(func):
        func = partial(func, func_name=name)
        FUNC_MAP[name] = func

        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return outer
