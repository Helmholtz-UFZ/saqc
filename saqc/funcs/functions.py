#! /usr/bin/env python
# -*- coding: utf-8 -*-


from ..dsl import evalExpression
from ..core.config import Params
from ..lib.tools import sesonalMask

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
    result = evalExpression(expression, flagger, data, flags, field, nodata=nodata)
    result = result.squeeze()

    if np.isscalar(result):
        raise TypeError(f"expression '{expression}' does not return an array")
    if not np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"expression '{expression}' does not return a boolean array")

    flags = flagger.setFlags(flags, field, result, **kwargs)
    return data, flags


@register("range")
def flagRange(data, flags, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flags = flagger.setFlags(flags, field, mask, **kwargs)
    return data, flags

@register("missing")
def flagMissing(data, flags, field, flagger, missing_indicator=np.nan, **kwargs):
    datacol = data[field].values

    if np.isnan(missing_indicator):
        mask = datacol.isna()
    else:
        mask = datacol[datacol == missing_indicator]

    flags = flagger.setFlags(flags, field, mask, **kwargs)
    return data, flags

@register('sesonalRange')
def flagSesonalRange(data, flags, field, flagger, min, max, startmonth=1, endmonth=12, startday=1, endday=31, **kwargs):
    smask = sesonalMask(flags.index, startmonth, startday, endmonth, endday)

    f = flags.loc[smask, [field]]
    d = data.loc[smask, [field]]
    if d.empty:
        return data, flags

    _, ff = flagRange(d, f.copy(), field, flagger, min=min, max=max, **kwargs)
    rangeflagged = flagger.getFlags(f[field]) != flagger.getFlags(ff[field])

    if rangeflagged.empty:
        return data, flags

    flags.update(ff.loc[rangeflagged, [field]])
    return data, flags


@register('clear')
def clearFlags(data, flags, field, flagger, **kwargs):
    flags = flagger.clearFlags(flags, field, **kwargs)
    return data, flags


@register('force')
def forceFlags(data, flags, field, flagger, **kwargs):
    flags = flagger.clearFlags(flags, field, **kwargs)
    flags = flagger.setFlags(flags, field, **kwargs)
    return data, flags

