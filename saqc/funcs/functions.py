#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from saqc.lib.tools import sesonalMask, flagWindow, groupConsecutives

from saqc.funcs.register import register


@register()
def procGeneric(data, field, flagger, func, **kwargs):
    # TODO:
    # - add new fields to te flagger
    data[field] = func.squeeze()
    return data, flagger


@register()
def flagGeneric(data, field, flagger, func, **kwargs):
    # NOTE:
    # The naming of the func parameter is pretty confusing
    # as it actually holds the result of a generic expression
    mask = func.squeeze()
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register()
def flagRange(data, field, flagger, min, max, **kwargs):
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register()
def flagMissing(data, field, flagger, nodata=np.nan, **kwargs):
    datacol = data[field]
    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol[datacol == nodata]

    flagger = flagger.setFlags(field, loc=mask, **kwargs)
    return data, flagger


@register()
def flagSesonalRange(
    data, field, flagger, min, max, startmonth=1, endmonth=12, startday=1, endday=31, **kwargs,
):
    smask = sesonalMask(data.index, startmonth, startday, endmonth, endday)

    d = data.loc[smask, [field]]
    if d.empty:
        return data, flagger

    _, flagger_range = flagRange(d, field, flagger.getFlagger(loc=d.index), min=min, max=max, **kwargs)

    if not flagger_range.isFlagged(field).any():
        return data, flagger

    flagger = flagger.setFlagger(flagger_range)
    return data, flagger


@register()
def clearFlags(data, field, flagger, **kwargs):
    flagger = flagger.clearFlags(field, **kwargs)
    return data, flagger


@register()
def forceFlags(data, field, flagger, flag, **kwargs):
    flagger = flagger.clearFlags(field).setFlags(field, flag=flag, **kwargs)
    return data, flagger


@register()
def flagIsolated(
    data, field, flagger, gap_window, group_window, **kwargs,
):

    gap_window = pd.tseries.frequencies.to_offset(gap_window)
    group_window = pd.tseries.frequencies.to_offset(group_window)

    col = data[field].mask(flagger.isFlagged(field))
    mask = col.isnull()

    flags = pd.Series(data=0, index=col.index, dtype=bool)
    for srs in groupConsecutives(mask):
        if np.all(~srs):
            start = srs.index[0]
            stop = srs.index[-1]
            if stop - start <= group_window:
                left = mask[start - gap_window : start].iloc[:-1]
                if left.all():
                    right = mask[stop : stop + gap_window].iloc[1:]
                    if right.all():
                        flags[start:stop] = True

    flagger = flagger.setFlags(field, flags, **kwargs)

    return data, flagger


@register()
def flagDummy(data, field, flagger, **kwargs):
    return data, flagger
