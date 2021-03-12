#!/usr/bin/env python


import numbers
import dios
import numpy as np
import pandas as pd
from typing import get_type_hints

from hypothesis.strategies import (
    lists,
    sampled_from,
    composite,
    from_regex,
    sampled_from,
    datetimes,
    integers,
    register_type_strategy,
    from_type,
)
from hypothesis.extra.numpy import arrays, from_dtype
from hypothesis.strategies._internal.types import _global_type_lookup

from saqc.common import *
from saqc.core.register import FUNC_MAP
from saqc.core.lib import SaQCFunction
from saqc.lib.types import FreqString, ColumnName, IntegerWindow
from saqc.flagger import Flagger, initFlagsLike

MAX_EXAMPLES = 50


# MAX_EXAMPLES = 100000


@composite
def dioses(draw, min_cols=1):
    """
    initialize data according to the current restrictions
    """
    # NOTE:
    # The following restriction showed up and should be enforced during init:
    # - Column names need to satisify the following regex: [A-Za-z0-9_-]+
    # - DatetimeIndex needs to be sorted
    # - Integer values larger than 2**53 lead to numerical instabilities during
    #   the integer->float->integer type conversion in _maskData/_unmaskData.

    cols = draw(lists(columnNames(), unique=True, min_size=min_cols))
    columns = {
        c: draw(dataSeries(min_size=3))
        for c in cols
    }
    return dios.DictOfSeries(columns)


@composite
def dataSeries(draw, min_size=0, max_size=100, dtypes=("float32", "float64", "int32", "int64")):
    if np.isscalar(dtypes):
        dtypes = (dtypes,)

    dtype = np.dtype(draw(sampled_from(dtypes)))
    if issubclass(dtype.type, numbers.Integral):
        info = np.iinfo(dtype)
    elif issubclass(dtype.type, numbers.Real):
        info = np.finfo(dtype)
    else:
        raise ValueError("only numerical dtypes are supported")
    # we don't want to fail just because of overflows
    elements = from_dtype(dtype, min_value=info.min + 1, max_value=info.max - 1)

    index = draw(daterangeIndexes(min_size=min_size, max_size=max_size))
    values = draw(arrays(dtype=dtype, elements=elements, shape=len(index)))
    return pd.Series(data=values, index=index)


@composite
def columnNames(draw):
    return draw(from_regex(r"[A-Za-z0-9_-]+", fullmatch=True))


@composite
def flaggers(draw, data):
    """
    initialize a flagger and set some flags
    """
    flagger = initFlagsLike(data)
    for col, srs in data.items():
        loc_st = lists(sampled_from(sorted(srs.index)), unique=True, max_size=len(srs) - 1)
        flagger[draw(loc_st), col] = BAD
    return flagger


@composite
def functions(draw, module: str = None):
    samples = tuple(FUNC_MAP.values())
    if module:
        samples = tuple(f for f in samples if f.name.startswith(module))
    # samples = [FUNC_MAP["drift.correctExponentialDrift"]]
    return draw(sampled_from(samples))


@composite
def daterangeIndexes(draw, min_size=0, max_size=100):
    min_date = pd.Timestamp("1900-01-01").to_pydatetime()
    max_date = pd.Timestamp("2099-12-31").to_pydatetime()
    start = draw(datetimes(min_value=min_date, max_value=max_date))
    periods = draw(integers(min_value=min_size, max_value=max_size))
    freq = draw(sampled_from(["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]))
    return pd.date_range(start, periods=periods, freq=freq)


@composite
def frequencyStrings(draw, _):
    freq = draw(sampled_from(["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]))
    mult = draw(integers(min_value=1, max_value=10))
    value = f"{mult}{freq}"
    return value


@composite
def dataFieldFlagger(draw):
    data = draw(dioses())
    field = draw(sampled_from(sorted(data.columns)))
    flagger = draw(flaggers(data))
    return data, field, flagger


@composite
def functionCalls(draw, module: str = None):
    func = draw(functions(module))
    kwargs = draw(functionKwargs(func))
    return func, kwargs


@composite
def functionKwargs(draw, func: SaQCFunction):
    data = draw(dioses())
    field = draw(sampled_from(sorted(data.columns)))

    kwargs = {
        "data": data,
        "field": field,
        "flagger": draw(flaggers(data))
    }

    column_name_strategy = lambda _: sampled_from(sorted(c for c in data.columns if c != field))
    interger_window_strategy = lambda _: integers(min_value=1, max_value=len(data[field]) - 1)

    register_type_strategy(FreqString, frequencyStrings)
    register_type_strategy(ColumnName, column_name_strategy)
    register_type_strategy(IntegerWindow, interger_window_strategy)

    for k, v in get_type_hints(func.func).items():
        if k not in {"data", "field", "flagger", "return"}:
            value = draw(from_type(v))
            # if v is TimestampColumnName:
            #     value = draw(columnNames())
            #     # we don't want to overwrite 'field'
            #     assume(value != field)
            #     # let's generate and add a timestamp column
            #     data[value] = draw(dataSeries(dtypes="datetime64[ns]", length=len(data[field])))
            #     # data[value] = draw(dataSeries(dtypes="datetime64[ns]"))
            kwargs[k] = value

    del _global_type_lookup[FreqString]
    del _global_type_lookup[ColumnName]
    del _global_type_lookup[IntegerWindow]

    return kwargs
