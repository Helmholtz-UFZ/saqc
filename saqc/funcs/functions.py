#! /usr/bin/env python
# -*- coding: utf-8 -*-


from ..dsl import evalExpression
from ..core.config import Params

from .register import register, FUNC_MAP

# keep all imports !
# to make all function register themself
from .break_detection import *
from .constants_detection import *
from .soil_moisture_tests import *
from .spike_detection import *
from .statistic_functions import *


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
