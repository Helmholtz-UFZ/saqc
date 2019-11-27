#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from saqc.lib.tools import sesonalMask, flagWindow

from saqc.funcs.register import register


@register("generic")
def flagGeneric(data, field, flagger, func, **kwargs):
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
    mask = func.squeeze() | flagger.isFlagged(flags_field)
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register("flagWindowAfterFlag")
def flagWindowAfterFlag(data, field, flagger, window, func, **kwargs):
    data, flagger_new = func
    flagger_repeated = flagWindow(
        flagger, flagger_new, field, direction="fw", window=window, **kwargs
    )
    return data, flagger_repeated


@register("flagNextAfterFlag")
def flagNextAfterFlag(data, field, flagger, n, func, **kwargs):
    data, flagger_new = func
    flagger_repeated = flagWindow(
        flagger, flagger_new, field, direction="fw", window=n, **kwargs
    )
    return data, flagger_repeated


@register("range")
def flagRange(data, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol >= max)
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register("missing")
def flagMissing(data, field, flagger, nodata=np.nan, **kwargs):
    datacol = data[field]
    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol[datacol == nodata]

    flagger = flagger.setFlags(field, loc=mask, **kwargs)
    return data, flagger


@register("sesonalRange")
def flagSesonalRange(
    data,
    field,
    flagger,
    min,
    max,
    startmonth=1,
    endmonth=12,
    startday=1,
    endday=31,
    **kwargs,
):
    smask = sesonalMask(data.index, startmonth, startday, endmonth, endday)

    d = data.loc[smask, [field]]
    if d.empty:
        return data, flagger

    _, flagger_range = flagRange(
        d, field, flagger.getFlagger(loc=d.index), min=min, max=max, **kwargs
    )

    if not flagger_range.isFlagged(field).any():
        return data, flagger

    flagger = flagger.setFlagger(flagger_range)
    return data, flagger


@register("clear")
def clearFlags(data, field, flagger, **kwargs):
    flagger = flagger.clearFlags(field, **kwargs)
    return data, flagger


@register("force")
def forceFlags(data, field, flagger, **kwargs):
    flagger = (flagger
               .clearFlags(field)
               .setFlags(field, **kwargs))
    return data, flagger


@register("isolated")
def flagIsolated(
    data,
    field,
    flagger,
    isolation_range,
    max_isolated_group_size=1,
    continuation_range="10min",
    drop_flags=None,
    **kwargs,
):

    drop_mask = pd.Series(data=False, index=data.index)
    # todo susp/BAD: make same case
    if drop_flags == "suspicious":
        drop_mask |= ~(flagger.isFlagged(field, flag=flagger.GOOD, comparator="<="))
    elif drop_flags == "BAD":
        drop_mask |= flagger.isFlagged(field, flag=flagger.BAD, comparator="==")
    elif isinstance(drop_flags, list):
        for to_drop in drop_flags:
            drop_mask |= flagger.isFlagged(field, flag=to_drop, comparator="==")

    dat_col = data[field][~drop_mask]
    dat_col.dropna(inplace=True)

    gap_check = dat_col.rolling(isolation_range).count()
    gap_check = gap_check[(gap_check.index[0] + pd.Timedelta(isolation_range)):]

    if max_isolated_group_size == 1:
        # isolated single values are much easier to identify:
        # exclude series initials:
        # reverse rolling trick:
        isolated_indices = gap_check[
            (gap_check[::-1].rolling(2).sum() == 2)[::-1].values
        ].index

    else:
        # check, which groups are centered enough for being isolated
        continuation_check = gap_check.rolling(continuation_range).count()
        # check which values are sparsely enough surrounded
        gap_check = (
            gap_check[::-1]
            .rolling(2)
            .apply(
                lambda x: int((x[0] == 1) & (x[1] <= max_isolated_group_size)),
                raw=False,
            )
        )
        gap_check = gap_check[::-1] == 1
        isolated_indices = gap_check[gap_check].index
        # check if the isolated values groups are sufficiently centered:
        isolated_indices = isolated_indices[
            continuation_check[isolated_indices] <= max_isolated_group_size
        ]
        # propagate True value onto entire isolated group
        # NOTE:
        # will not work with bfill method, because its not sure the frequency
        # grid is actually equidistant - so here comes the rolling reverse
        # trick for offset defined windows again
        gap_check[:] = np.nan
        gap_check.loc[isolated_indices] = True
        original_index = gap_check.index
        gap_check = gap_check[::-1]
        pseudo_increasing_index = gap_check.index[0] - gap_check.index
        gap_check.index = pseudo_increasing_index
        gap_check = gap_check.rolling(continuation_range).count().notna()[::-1]
        isolated_indices = original_index[gap_check.values]

    flagger = flagger.setFlags(field, isolated_indices, **kwargs)

    return data, flagger
