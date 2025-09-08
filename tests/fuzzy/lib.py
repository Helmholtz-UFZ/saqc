#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numbers
from contextlib import contextmanager
from typing import get_type_hints

import numpy as np
import pandas as pd
from hypothesis.extra.numpy import arrays, from_dtype
from hypothesis.strategies import (
    composite,
    datetimes,
    from_regex,
    from_type,
    integers,
    lists,
    register_type_strategy,
    sampled_from,
)
from hypothesis.strategies._internal.types import _global_type_lookup

from saqc import BAD, DictOfSeries
from saqc.core.flags import initFlagsLike
from saqc.core.register import FUNC_MAP

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
    columns = {c: draw(dataSeries(min_size=3)) for c in cols}
    return DictOfSeries(columns)


@composite
def dataSeries(
    draw, min_size=0, max_size=100, dtypes=("float32", "float64", "int32", "int64")
):
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
def flagses(draw, data):
    """
    initialize a flags and set some flags
    """
    flags = initFlagsLike(data)
    for col, srs in data.items():
        loc_st = lists(
            sampled_from(sorted(srs.index)), unique=True, max_size=len(srs) - 1
        )
        flags[draw(loc_st), col] = BAD
    return flags


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
    freq = draw(sampled_from(["D", "h", "min", "s", "ms", "us", "ns"]))
    return pd.date_range(start, periods=periods, freq=freq)


@composite
def frequencyStrings(draw, _):
    freq = draw(sampled_from(["D", "h", "min", "s", "ms", "us", "ns"]))
    mult = draw(integers(min_value=1, max_value=10))
    value = f"{mult}{freq}"
    return value


@composite
def dataFieldFlags(draw):
    # Draw the base data
    data = draw(dioses())

    # Pick a column
    field = draw(sampled_from(sorted(data.columns)))

    # Draw flags
    flags = draw(flagses(data))

    # Draw start_date and end_date from the index
    index_dates = sorted(data[field].index)

    # Only proceed if there are at least 2 dates
    if len(index_dates) >= 2:
        start_date = draw(sampled_from(index_dates[:-1]))  # exclude last so start < end
        # Restrict end_date to dates after start_date
        possible_end_dates = [d for d in index_dates if d > start_date]
        end_date = draw(sampled_from(possible_end_dates))
    else:
        # fallback if there are not enough dates
        start_date = end_date = index_dates[0]

    return data, field, flags, start_date, end_date


@composite
def functionCalls(draw, module: str = None):
    func = draw(functions(module))
    kwargs = draw(functionKwargs(func))
    return func, kwargs


@contextmanager
def applyStrategies(strategies: dict):
    for dtype, strategy in strategies.items():
        register_type_strategy(dtype, strategy)

    yield

    for dtype in strategies.keys():
        del _global_type_lookup[dtype]


@composite
def functionKwargs(draw, func):
    data = draw(dioses())
    field = draw(sampled_from(sorted(data.columns)))

    kwargs = {"data": data, "field": field, "flags": draw(flagses(data))}

    i64 = np.iinfo("int64")

    strategies = {
        # FreqString: frequencyStrings,
        # ColumnName: lambda _: sampled_from(
        #     sorted(c for c in data.columns if c != field)
        # ),
        # IntegerWindow: lambda _: integers(min_value=1, max_value=len(data[field]) - 1),
        int: lambda _: integers(min_value=i64.min + 1, max_value=i64.max - 1),
    }

    with applyStrategies(strategies):
        for k, v in get_type_hints(func).items():
            if k not in {"data", "field", "flags", "return"}:
                value = draw(from_type(v))
                kwargs[k] = value

    return kwargs
