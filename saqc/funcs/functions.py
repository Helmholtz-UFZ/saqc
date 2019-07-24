#! /usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np

from ..dsl import evalExpression
from ..core.config import Params

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


def flagDispatch(func_name, *args, **kwargs):
    func = FUNC_MAP.get(func_name, None)
    if func is not None:
        return func(*args, **kwargs)
    raise NameError(f"function name {func_name} is not definied")


@register("generic")
def flagGeneric(data, flags, field, flagger, nodata=np.nan, **kwargs):
    expression = kwargs[Params.FUNC]
    result = evalExpression(expression, flagger,
                            data, flags, field,
                            nodata=nodata)

    result = result.squeeze()

    if np.isscalar(result):
        raise TypeError(f"expression '{expression}' does not return an array")

    if not np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"expression '{expression}' does not return a boolean array")

    fchunk = flagger.setFlag(flags=flags.loc[result, field], **kwargs)

    flags.loc[result, field] = fchunk

    return data, flags


@register("range")
def flagRange(data, flags, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flags.loc[mask, field] = flagger.setFlag(flags.loc[mask, field], **kwargs)
    return data, flags
