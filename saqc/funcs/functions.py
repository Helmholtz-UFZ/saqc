#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..lib.tools import sesonalMask, flagWindow

from .register import register

# def flagDispatch(func_name, *args, **kwargs):
#     func = FUNC_MAP.get(func_name, None)
#     if func is not None:
#         return func(*args, **kwargs)
#     raise NameError(f"function name {func_name} is not definied")


@register("generic")
def flagGeneric(data, flags, field, flagger, func, **kwargs):
    # NOTE:
    # - The naming of the func parameter is pretty confusing
    #   as it actually holds the result of a generic expression
    # - if the result series carries a name, it was explicitly created
    #   from one single columns, so we need to preserve this columns
    #   properties
    # - the check if func.name is in data.columns is necessary as
    #   DmpFlagger.isFlagged does not preserve the name of the column
    #   it was executed on -> would be nice to overcome this restriction
    flags_field = func.name if func.name in data.columns else field
    mask = func.squeeze() | flagger.isFlagged(flags, flags_field)
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")
    flags = flagger.setFlags(flags, field, mask, **kwargs)
    return data, flags


@register("flagWindowAfterFlag")
def flagWindowAfterFlag(data, flags, field, flagger, window, func, **kwargs):
    data, new_flags = func
    repeated_flags = flagWindow(flags, new_flags,
                                field, flagger,
                                direction='fw', window=window,
                                **kwargs)
    return data, repeated_flags


@register("flagNextAfterFlag")
def flagNextAfterFlag(data, flags, field, flagger, n, func, **kwargs):
    data, new_flags = func
    repeated_flags = flagWindow(flags, new_flags,
                                field, flagger,
                                direction='fw', window=n,
                                **kwargs)
    return data, repeated_flags


@register("range")
def flagRange(data, flags, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flags = flagger.setFlags(flags, field, mask, **kwargs)
    return data, flags


@register("missing")
def flagMissing(data, flags, field, flagger, nodata=np.nan, **kwargs):

    datacol = data[field]

    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol[datacol == nodata]

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
    rangeflagged = flagger.getFlags(f, field) != flagger.getFlags(ff, field)

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

@register('isolated')
def flagIsolated(data, flags, field, flagger, isolation_range, max_isolated_group_size=1, continuation_range='10min',
                 drop_flags=None, **kwargs):

    drop_mask = pd.Series(data=False, index=flags.index)
    # todo susp/BAD: make same case
    if drop_flags is 'suspicious':
        drop_mask |= ~(flagger.isFlagged(flags, field, flag=flagger.GOOD, comparator='<='))
    elif drop_flags is 'BAD':
        drop_mask |= flagger.isFlagged(flags, field, flag=flagger.BAD, comparator='==')
    elif isinstance(drop_flags, list):
        for to_drop in drop_flags:
            drop_mask |= flagger.isFlagged(flags, field, flag=to_drop, comparator='==')

    dat_col = data[field][~drop_mask]
    dat_col.dropna(inplace=True)

    if max_isolated_group_size == 1:
        # isolated single values are much easier to identify:
        gap_check = dat_col.rolling(isolation_range).count()
        # exclude series initials:
        gap_check = gap_check[(gap_check.index[0] + pd.Timedelta(isolation_range)):]
        # reverse rolling trick:
        isolated_indices = gap_check[(gap_check[::-1].rolling(2).sum() == 2)[::-1].values].index

    else:
        gap_check = dat_col.rolling(isolation_range).count()
        # check, which groups are centered enough for being isolated
        continuation_check = gap_check.rolling(continuation_range).count()
        # exclude series initials:
        gap_check = gap_check[(gap_check.index[0] + pd.Timedelta(isolation_range)):]
        # check which values are sparsely enough surrounded
        gap_check = gap_check[::-1].rolling(2).apply(lambda x: int((x[0] == 1) & (x[1] <= max_isolated_group_size)),
                                                     raw=False)
        gap_check = (gap_check[::-1] == 1)
        isolated_indices = gap_check[gap_check].index
        # check if the isolated values groups are sufficiently centered:
        isolated_indices = isolated_indices[continuation_check[isolated_indices] <= max_isolated_group_size]
        # propagate True value onto entire isolated group (will not work with bfill method, because its not sure the
        # frequencie grid is actually equidistant - so here comes rolling reverse trick for offset defined windows
        # again):
        gap_check[:] = np.nan
        gap_check.loc[isolated_indices] = True
        original_index = gap_check.index
        gap_check = gap_check[::-1]
        pseudo_increasing_index = gap_check.index[0] - gap_check.index
        gap_check.index = pseudo_increasing_index
        gap_check = gap_check.rolling(continuation_range).count().notna()[::-1]
        isolated_indices = original_index[gap_check.values]

    flags = flagger.setFlags(flags, field, isolated_indices, **kwargs)

    return data, flags
