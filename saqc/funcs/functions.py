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
def flagPattern(data, field, flagger, ref_datafield, sample_freq = '15 Min', method = 'dtw', min_distance = None, **kwargs):
    # To Do: Bisher geht der Algo davon aus, dass sich das Pattern über einen Tag erstreckt -> noch erweitern auf > 1 Tag bzw. < 1 Tag (z.B. alle zwei Stunden ein Pattern)

    test_field = data[field].copy()
    ref_field  = data[ref_datafield].copy()
    # Test : Referenzhut ist der 30. Juni 2019
    ref_field = ref_field['2019-06-30 11:00:00':'2019-06-30 18:00:00']

    # Get start and end time from reference field
    Start_date = ref_field.head(1).index.item()
    Start_time = datetime.datetime.time(Start_date)
    End_date = ref_field.tail(1).index.item()
    End_time = datetime.datetime.time(End_date)

    # Count length of partition
    partition_count = End_date.day - Start_date.day + 1
    partition_freq = "%d D" % partition_count

    # Harmonization
    test_field = test_field.resample(sample_freq).first().interpolate('time')
    ref_field = ref_field.resample(sample_freq).first().interpolate('time')




    # Calculate Partition
#    if not partition_freq:
#        partition_freq = test_field.shape[0]

#    if isinstance(partition_freq, str):

    partitions = test_field.groupby(pd.Grouper(freq=partition_freq))

#    else:
#        grouper_series = pd.Series(data=np.arange(0, test_field.shape[0]), index=test_field.index)
#        grouper_series = grouper_series.transform(lambda x: int(np.floor(x / partition_freq)))
#        partitions = test_field.groupby(grouper_series)

    # Initialize the chosen pattern method
    # DTW as standard
    if (not method) or (method != 'dtw' and method != 'wavelet'):
        method = 'dtw'
    # Initializing DTW
    if method == 'dtw':
        # Set minimal distance
        if not min_distance:
            min_distance = 4.5
    # Initializing Wavelets
    if method == 'wavelet':
        # calculate reference wavelet transform
        ref_field_wl = ref_field.values.ravel()
        # Widths lambda as in Ann Maharaj
        widths = [1, 2, 4, 8]
        cwtmat_ref, freqs = pywt.cwt(ref_field_wl, widths, 'mexh')
        # Square of matrix elements as Power sum of the matrix
        wavepower_ref = np.power(cwtmat_ref, 2)



    flags = pd.Series(data=0, index=test_field.index)
    # calculate flags for every partition
    partition_min = 0
    # Set time frames for partition
    if not Start_time:
        Start_time = '00:00'
    if not End_time:
        End_time = '23:59'
    for _, partition in partitions:
        if partition.empty | (partition.shape[0] < partition_min):
            continue
        # Use only the given time frame
        pattern = partition.between_time(Start_time, End_time)
        # Choose method
        if method == 'dtw':
            distance = dtw.dtw(pattern, ref_field, open_end=True, distance_only=True).normalizedDistance
            # Partition labeled as pattern by dtw
            if distance < min_distance:
                flags[partition.index] = 1
        elif method == 'wavelet':
            # calculate reference wavelet transform
            test_field_wl = pattern.values.ravel()
            cwtmat_test, freqs = pywt.cwt(test_field_wl, widths, 'mexh')
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
