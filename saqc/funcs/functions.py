#! /usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import pandas as pd

from saqc.lib.tools import groupConsecutives, sesonalMask

from saqc.core.register import register, Func
from saqc.core.visitor import ENVIRONMENT


def _dslIsFlagged(flagger, var, flag=None, comparator=None):
    """
    helper function for `flagGeneric`
    """
    if comparator is None:
        return flagger.isFlagged(var.name, flag=flag)
    return flagger.isFlagged(var.name, flag=flag, comparator=comparator)


def _execGeneric(flagger, data, func, field, nodata):

    # TODO:
    # - check series.index compatibility
    # - field is only needed to translate 'this' parameters
    #    -> maybe we could do the translation on the tree instead

    func = Func(func)
    for k in func.parameters:
        k = field if k == "this" else k
        if k not in data:
            raise NameError(f"variable '{k}' not found")
        func = Func(func, data[k])

    globs = {
        "isflagged": partial(_dslIsFlagged, flagger),
        "ismissing": lambda var: ((var == nodata) | pd.isnull(var)),
        "this": field,
        "NODATA": nodata,
        "GOOD": flagger.GOOD,
        "BAD": flagger.BAD,
        "UNFLAGGED": flagger.UNFLAGGED,
        **ENVIRONMENT
    }
    func = func.addGlobals(globs)
    return func()


@register
def procGeneric(data, field, flagger, func, nodata=np.nan, **kwargs):
    """
    Execute generic functions.
    The **kwargs are needed to satisfy the test-function interface,
    although they are of no use here. Usually they are abused to
    transport the name of the test function (here: `procGeneric`)
    into the flagger, but as we don't set flags here, we simply
    ignore them
    """
    data[field] = _execGeneric(flagger, data, func, field, nodata).squeeze()
    # NOTE:
    # The flags to `field` will be (re-)set to UNFLAGGED
    # That leads to the following problem:
    # flagger.merge merges the given flaggers, if
    # `field` did already exist before the call to `procGeneric`
    # but with a differing index, we end up with:
    # len(data[field]) != len(flagger.getFlags(field))
    # see: test/funcs/test_generic_functions.py::test_procGenericMultiple

    # TODO:
    # We need a way to simply overwrite a given flagger column, maybe
    # an optional keyword to merge ?
    flagger = flagger.merge(flagger.initFlags(data[field]))
    return data, flagger


@register
def flagGeneric(data, field, flagger, func, nodata=np.nan, **kwargs):
    # NOTE:
    # The naming of the func parameter is pretty confusing
    # as it actually holds the result of a generic expression
    mask = _execGeneric(flagger, data, func, field, nodata).squeeze()
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")

    if flagger.getFlags(field).empty:
        flagger = flagger.merge(flagger.initFlags(data=pd.Series(name=field, index=mask.index)))
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register
def flagRange(data, field, flagger, min, max, **kwargs):
    # using .values is very much faster
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register
def flagMissing(data, field, flagger, nodata=np.nan, **kwargs):
    datacol = data[field]
    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol[datacol == nodata]

    flagger = flagger.setFlags(field, loc=mask, **kwargs)
    return data, flagger


@register
def flagSesonalRange(
    data, field, flagger, min, max, startmonth=1, endmonth=12, startday=1, endday=31, **kwargs,
):
    smask = sesonalMask(data[field].index, startmonth, startday, endmonth, endday)

    d = data.loc[smask, [field]]
    if d.empty:
        return data, flagger

    _, flagger_range = flagRange(d, field, flagger.slice(loc=d[field].index), min=min, max=max, **kwargs)

    if not flagger_range.isFlagged(field).any():
        return data, flagger

    flagger = flagger.merge(flagger_range)
    return data, flagger


@register
def clearFlags(data, field, flagger, **kwargs):
    flagger = flagger.clearFlags(field, **kwargs)
    return data, flagger


@register
def forceFlags(data, field, flagger, flag, **kwargs):
    flagger = flagger.clearFlags(field).setFlags(field, flag=flag, **kwargs)
    return data, flagger


@register
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


@register
def flagDummy(data, field, flagger, **kwargs):
    """ Do nothing """  
    return data, flagger


@register
def flagCrossScoring(data, field, flagger, fields, thresh, cross_stat=np.median, **kwargs):
    val_frame = data[fields].loc[data[fields].index_of('shared')].to_df()
    if not isinstance(cross_stat,str):
        try:
            stat = getattr(val_frame, cross_stat.__name__)(axis=1)
        except AttributeError:
            stat = val_frame.aggregate(cross_stat, axis=1)
        diff_scores = val_frame.subtract(stat, axis=0).abs()

    else:
        if cross_stat == 'modZscore':
            MAD_series = val_frame.subtract(val_frame.median(axis=1), axis=0).abs().median(axis=1)
            diff_scores = ((0.6745 * (val_frame.subtract(val_frame.median(axis=1), axis=0)))
                           .divide(MAD_series, axis=0)).abs()
        if cross_stat == 'Zscore':
            diff_scores = (val_frame.subtract(val_frame.mean(axis=1), axis=0)).divide(val_frame.std(axis=1), axis=0).abs()

    diff_scores = diff_scores > thresh
    for var in fields:
        flagger = flagger.setFlags(var, diff_scores[var].values, **kwargs)
    return data, flagger
