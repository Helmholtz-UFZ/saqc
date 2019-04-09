#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from lib.tools import valueRange, slidingWindowIndices
from dsl import evalExpression
from config import Params


def flagDispatch(func_name, *args, **kwargs):
    func_map = {
        "manflag": flagManual,
        "mad": flagMad,
        "constant": flagConstant,
        "generic": flagGeneric}

    func = func_map.get(func_name, None)
    if func is not None:
        return func(*args, **kwargs)
    raise NameError(f"function name {func_name} is not definied")



def flagGeneric(data, flags, field, flagger, nodata=np.nan, **flag_params):

    expression = flag_params[Params.FUNC]
    result = evalExpression(expression, flagger,
                            data, flags, field,
                            nodata=nodata)

    if np.isscalar(result):
        raise TypeError(f"expression '{expression}' does not return an array")

    if not np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"expression '{expression}' does not return a boolean array")

    to_flag = result.filled(True)

    fchunk = flagger.setFlag(flags=flags.loc[to_flag, field], **flag_params)

    # .values transform the data to NumPy array, so the assignment
    # can work for single- and multi- indexed data.
    flags.loc[to_flag, field] = fchunk.values

    return data, flags


def flagConstant(data, flags, field, flagger, eps,
                 length, thmin=None, **kwargs):

    datacol = data[field]
    flagcol = flags[field]

    length = ((pd.to_timedelta(length) - data.index.freq)
              .to_timedelta64()
              .astype(np.int64))

    values = (datacol
              .mask((datacol < thmin) | datacol.isnull())
              .values
              .astype(np.int64))

    dates = datacol.index.values.astype(np.int64)

    mask = np.isfinite(values)

    for start_idx, end_idx in slidingWindowIndices(datacol.index, length):
        mask_chunk = mask[start_idx:end_idx]
        values_chunk = values[start_idx:end_idx][mask_chunk]
        dates_chunk = dates[start_idx:end_idx][mask_chunk]

        # we might have removed dates from the start/end of the
        # chunk resulting in a period shorter than 'length'
        # print (start_idx, end_idx)
        if valueRange(dates_chunk) < length:
            continue
        if valueRange(values_chunk) < eps:
            flagcol[start_idx:end_idx] = flagger.setFlags(flagcol[start_idx:end_idx], **kwargs)

    data[field] = datacol
    flags[field] = flagcol
    return data, flags


def flagManual(data, flags, field, flagger, **kwargs):
    return data, flags


def flagMad(data, flags, field, flagger, length, z, deriv, **kwargs):

    def _flagMad(data: np.ndarray, z: int, deriv: int) -> np.ndarray:
        # NOTE: numpy is at least twice as fast as numba.jit(nopython)
        # median absolute deviation
        for i in range(deriv):
            data[i+1:] = np.diff(data[i:])
            data[i] = np.nan
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data-median))
        tresh = mad * (z/0.6745)
        with np.errstate(invalid="ignore"):
            return (data < (median - tresh)) | (data > (median + tresh))

    datacol = data[field]
    flagcol = flags[field]

    values = (datacol
              .mask(flagger.isFlagged(flagcol))
              .values)

    window = pd.to_timedelta(length) - pd.to_timedelta(data.index.freq)
    mask = np.zeros_like(values, dtype=bool)

    for start_idx, end_idx in slidingWindowIndices(datacol.index, window, "1D"):
        mad_flags = _flagMad(values[start_idx:end_idx], z, deriv)
        # reset the mask
        mask[:] = False
        mask[start_idx:end_idx] = mad_flags
        flagcol[mask] = flagger.setFlag(flagcol[mask], **kwargs)

    flags[field] = flagcol

    return data, flags
