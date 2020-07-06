#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dtw
import pywt
from mlxtend.evaluate import permutation_test
import datetime

from saqc.lib.tools import groupConsecutives, sesonalMask

from saqc.funcs.register import register


@register()
def procGeneric(data, field, flagger, func, **kwargs):
    data[field] = func.squeeze()
    # NOTE:
    # The flags to `field` will be (re-)set to UNFLAGGED

    # PROBLEM:
    # flagger.setFlagger merges the given flaggers, if
    # `field` did already exist before the call to `procGeneric`
    # but with a differing index, we end up with:
    # len(data[field]) != len(flagger.getFlags(field))
    # see: test/funcs/test_generic_functions.py::test_procGenericMultiple
    flagger = flagger.setFlagger(flagger.initFlags(data[field]))
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

    if flagger.getFlags(field).empty:
        flagger = flagger.setFlagger(flagger.initFlags(data=pd.Series(name=field, index=mask.index)))
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register()
def flagRange(data, field, flagger, min, max, **kwargs):
    # using .values is very much faster
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger




@register()
def flagPattern(data, field, flagger, reference_field, method = 'dtw', partition_freq = "days", partition_offset = 0, max_distance = 0.03, normalized_distance = True, widths = [1,2,4,8], waveform = 'mexh', **kwargs):

    test = data[field].copy()
    ref = data[reference_field].copy()
    #ref = ref['2019-06-29 11:00:00':'2019-06-29 18:00:00']
    Pattern_start_date = ref.index[0]
    Pattern_start_time = datetime.datetime.time(Pattern_start_date)
    Pattern_end_date = ref.index[-1]
    Pattern_end_time = datetime.datetime.time(Pattern_end_date)

    ### Extract partition frequency from pattern if needed
    if not isinstance(partition_freq, str):
        raise ValueError('Partition frequency has to be given in string format.')
    elif partition_freq == "days" or partition_freq == "months":
            # Get partition frequency from reference field
            partition_count = (Pattern_end_date - Pattern_start_date).days
            partitions = test.groupby(pd.Grouper(freq="%d D" % (partition_count + 1)))
    else:
        partitions = test.groupby(pd.Grouper(freq=partition_freq))

    # Initializing Wavelets
    if method == 'wavelet':
        # calculate reference wavelet transform
        ref_wl = ref.values.ravel()
        # Widths lambda as in Ann Maharaj
        cwtmat_ref, freqs = pywt.cwt(ref_wl, widths, waveform)
        # Square of matrix elements as Power sum of the matrix
        wavepower_ref = np.power(cwtmat_ref, 2)
    elif not method == 'dtw':
    # No correct method given
        raise ValueError('Unable to interpret {} as method.'.format(method))

    flags = pd.Series(data=0, index=test.index)
    ### Calculate flags for every partition
    partition_min = ref.shape[0]
    for _, partition in partitions:
        # Ensuring that partition is at least as long as reference pattern
        if partition.empty or (partition.shape[0] < partition_min):
            continue
        if partition_freq == "days" or partition_freq == "months":
            # Use only the time frame given by the pattern
            test = partition.between_time(Pattern_start_time, Pattern_end_time)
            mask = (partition.index >= test.index[0]) & (partition.index <= test.index[-1])
            test = partition.loc[mask]
        else:
            # cut partition according to pattern and offset
            start_time = pd.Timedelta(partition_offset) + partition.index[0]
            end_time = start_time + pd.Timedelta(Pattern_end_date - Pattern_start_date)
            test = partition[start_time:end_time]
        ### Switch method
        if method == 'dtw':
            distance = dtw.dtw(test, ref, open_end=True, distance_only=True).normalizedDistance
            if normalized_distance:
                distance = distance/abs(ref.mean())
            # Partition labeled as pattern by dtw
            if distance < max_distance:
                flags[partition.index] = 1
        elif method == 'wavelet':
            # calculate reference wavelet transform
            test_wl = test.values.ravel()
            cwtmat_test, freqs = pywt.cwt(test_wl, widths, 'mexh')
            # Square of matrix elements as Power sum of the matrix
            wavepower_test = np.power(cwtmat_test, 2)
            # Permutation test on Powersum of matrix
            p_value = []
            for i in range(len(widths)):
                x = wavepower_ref[i]
                y = wavepower_test[i]
                pval = permutation_test(x, y, method='approximate', num_rounds=200, func=lambda x, y: x.sum() / y.sum(),
                                        seed=0)
                p_value.append(min(pval, 1 - pval))
            # Partition labeled as pattern by wavelet
            if min(p_value) >= 0.01:
                flags[partition.index] = 1

    mask = (flags == 1)

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
    smask = sesonalMask(data[field].index, startmonth, startday, endmonth, endday)

    d = data.loc[smask, [field]]
    if d.empty:
        return data, flagger

    _, flagger_range = flagRange(d, field, flagger.getFlagger(loc=d[field].index), min=min, max=max, **kwargs)

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
