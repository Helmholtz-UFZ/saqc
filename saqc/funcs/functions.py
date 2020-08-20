#! /usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import pandas as pd
import dtw
import pywt
from mlxtend.evaluate import permutation_test
import datetime

from saqc.lib.tools import groupConsecutives, seasonalMask
from saqc.funcs.proc_functions import proc_fork, proc_drop, proc_projectFlags
from saqc.funcs.data_modelling import modelling_mask

from saqc.core.register import register, Func
from saqc.core.visitor import ENVIRONMENT
from dios import DictOfSeries
from typing import Any


def _dslIsFlagged(flagger, var, flag=None, comparator=">="):
    """
    helper function for `flagGeneric`
    """
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
        **ENVIRONMENT,
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
def flagPattern(data, field, flagger, reference_field, method = 'dtw', partition_freq = "days", partition_offset = 0, max_distance = 0.03, normalized_distance = True, open_end = True, widths = (1, 2, 4, 8), waveform = 'mexh', **kwargs):
    """ Implementation of two pattern recognition algorithms:
    - Dynamic Time Warping (dtw) [1]
    - Pattern recognition via wavelets [2]

    The steps are:
    1. Get the frequency of partitions, in which the time series has to be divided (for example: a pattern occurs daily, or every hour)
    2. Compare each partition with the given pattern
    3. Check if the compared partition contains the pattern or not
    4. Flag partition if it contains the pattern

    :param data:                pandas dataframe. holding the data
    :param field:               fieldname in `data`, which holds the series to be checked for patterns
    :param flagger:             flagger.
    :param reference_field:     fieldname in `data`, which holds the pattern
    :param method:              str. Pattern Recognition method to be used: 'dtw' or 'wavelets'. Default: 'dtw'
    :param partition_freq:      str. Frequency, in which the pattern occurs. If only "days" or "months" is given, then precise length of partition is calculated from pattern length. Default: "days"
    :param partition_offset:    str. If partition frequency is given, and pattern starts after a timely offset (e.g., partition frequency is "1 h", pattern starts at 10:15, then offset is "15 min"). Default: 0
    :param max_distance:        float. For dtw. Maximum dtw-distance between partition and pattern, so that partition is recognized as pattern. Default: 0.03
    :param normalized_distance: boolean. For dtw. Normalizing dtw-distance (see [1]). Default: True
    :param open_end:            boolean. For dtw. End of pattern is matched with a value in the partition (not necessarily end of partition). Recommendation of [1]. Default: True
    :param widths:              tuple of int. For wavelets. Widths for wavelet decomposition. [2] recommends a dyadic scale. Default: (1,2,4,8)
    :param waveform:            str. For wavelets. Wavelet to be used for decomposition. Default: 'mexh'

    Literature:
    [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
    [2] Maharaj, E.A. (2002): Pattern Recognition of Time Series using Wavelets. In: Härdle W., Rönz B. (eds) Compstat. Physica, Heidelberg, 978-3-7908-1517-7.
    """

    test = data[field].copy()
    ref = data[reference_field].copy()
    pattern_start_date = ref.index[0].time()
    pattern_end_date = ref.index[-1].time()

    ### Extract partition frequency from pattern if needed
    if not isinstance(partition_freq, str):
        raise ValueError('Partition frequency has to be given in string format.')
    elif partition_freq == "days" or partition_freq == "months":
            # Get partition frequency from reference field
            partition_count = (pattern_end_date - pattern_start_date).days
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

    flags = pd.Series(data=False, index=test.index)
    ### Calculate flags for every partition
    partition_min = ref.shape[0]
    for _, partition in partitions:

        # Ensuring that partition is at least as long as reference pattern
        if partition.empty or (partition.shape[0] < partition_min):
            continue
        if partition_freq == "days" or partition_freq == "months":
            # Use only the time frame given by the pattern
            test = partition[pattern_start_date:pattern_start_date]
            mask = (partition.index >= test.index[0]) & (partition.index <= test.index[-1])
            test = partition.loc[mask]
        else:
            # cut partition according to pattern and offset
            start_time = pd.Timedelta(partition_offset) + partition.index[0]
            end_time = start_time + pd.Timedelta(pattern_end_date - pattern_start_date)
            test = partition[start_time:end_time]
        ### Switch method
        if method == 'dtw':
            distance = dtw.dtw(test, ref, open_end = open_end, distance_only = True).normalizedDistance
            if normalized_distance:
                distance = distance/ref.var()
            # Partition labeled as pattern by dtw
            if distance < max_distance:
                flags[partition.index] = True
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
                flags[partition.index] = True


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

    data, flagger = proc_fork(data, field, flagger, suffix="_masked")
    data, flagger = modelling_mask(data, field + "_masked", flagger, mode='seasonal',
                                   season_start=str(startmonth).zfill(2) + '-' + str(startday).zfill(2) + 'T00:00:00',
                                   season_end=str(endmonth).zfill(2) + '-' + str(endday).zfill(2) + 'T00:00:00',
                                   inclusive_selection='season')
    data, flagger = flagRange(data, field + "_masked", flagger, min=min, max=max, **kwargs)
    data, flagger = proc_projectFlags(data, field, flagger, method='match', source=field + "_masked")
    data, flagger = proc_drop(data, field + "_masked", flagger)
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
def flagForceFail(data, field, flagger, **kwargs):
    """ Raise a RuntimeError. """
    raise RuntimeError("Works as expected :D")


@register
def flagUnflagged(data, field, flagger, **kwargs):
    flag = kwargs.pop('flag', flagger.GOOD)
    flagger = flagger.setFlags(field, flag=flag, **kwargs)
    return data, flagger


@register
def flagGood(data, field, flagger, **kwargs):
    kwargs.pop('flag', None)
    return flagUnflagged(data, field, flagger)


@register
def flagManual(data, field, flagger, mdata, mflag: Any = 1, method="plain", **kwargs):
    """ Flag data by given manual data.

    The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
    The format of mdata can be a indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
    but also can be a plain list- or array-like.
    How indexed mdata is aligned to data is specified via `method` argument.

    Parameters
    ----------
    data : DictOfSeries
    field : str
        The field chooses the column in flags and data in question.
        It also determine the column in mdata if its of type pd.Dataframe or dios.DictOfSeries.

    flagger : flagger
range_dict.keys()
    mdata : {pd.Series, pd.Dataframe, DictOfSeries, str}
        The manual data

    mflag : scalar
        The flag that indicates data points in `mdata`, that should be flagged.

    method : {'plain', 'ontime', 'left-open', 'right-open'}, default plain
        Define how mdata is applied on data. Except 'plain' mdata must have a index.
        * 'plain': mdata must have same length than data and is applied one-to-one on data.
        * 'ontime': work only with indexed mdata, it is applied, where timestamps are match.
        * 'right-open': mdata defines periods, which are defined by two consecutive timestamps, the
            value of the first aka. left is applied on the whole period.
        * 'left-open': like 'right-open' but the value is defined in the latter aka. right timestamp.

    kwargs : Any
        passed to flagger

    Returns
    -------
    data, flagger: original data, modified flagger
    
    Examples
    --------
    An example for mdata
    >>> mdata = pd.Series([1,0,1], index=pd.to_datetime(['2000-02', '2000-03', '2001-05']))
    >>> mdata
    2000-02-01    1
    2000-03-01    0
    2001-05-01    1
    dtype: int64

    On *dayly* data, with the 'ontime' method, only the provided timestamnps are used.
    Bear in mind that only exact timestamps apply, any offset will result in ignoring
    the timestamp.
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='ontime')
    >>> fl.isFlagged(field)
    2000-01-31    False
    2000-02-01    True
    2000-02-02    False
    2000-02-03    False
    ..            ..
    2000-02-29    False
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool

    With the 'right-open' method, the mdata is forward fill:
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='right-open')
    >>> fl.isFlagged(field)
    2000-01-31    False
    2000-02-01    True
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    False
    2000-03-02    False
    Freq: D, dtype: bool

    With the 'left-open' method, backward filling is used:
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='left-open')
    >>> fl.isFlagged(field)
    2000-01-31    False
    2000-02-01    False
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool
    """
    dat = data[field]
    if isinstance(mdata, str):
        # todo import path type in mdata, use
        #  s = pd.read_csv(mdata, index_col=N, usecol=[N,N,..]) <- use positional
        #  use a list-arg in config to get the columns
        #  at last, fall throug to next checks
        raise NotImplementedError("giving a path is currently not supported")

    if isinstance(mdata, (pd.DataFrame, DictOfSeries)):
        mdata = mdata[field]

    hasindex = isinstance(mdata, (pd.Series, pd.DataFrame, DictOfSeries))
    if not hasindex and method != "plain":
        raise ValueError("mdata has no index")

    if method == "plain":
        if hasindex:
            mdata = mdata.to_numpy()
        if len(mdata) != len(dat):
            raise ValueError("mdata must have same length then data")
        mdata = pd.Series(mdata, index=dat.index)
    elif method == "ontime":
        pass  # reindex will do the job later
    elif method in ["left-open", "right-open"]:
        mdata = mdata.reindex(dat.index.union(mdata.index))

        # -->)[t0-->)[t1--> (ffill)
        if method == "right-open":
            mdata = mdata.ffill()

        # <--t0](<--t1](<-- (bfill)
        if method == "left-open":
            mdata = mdata.bfill()
    else:
        raise ValueError(method)

    mask = mdata == mflag
    mask = mask.reindex(dat.index).fillna(False)
    flagger = flagger.setFlags(field=field, loc=mask, **kwargs)
    return data, flagger


@register
def flagCrossScoring(data, field, flagger, fields, thresh, cross_stat=np.median, **kwargs):
    df = data[fields].loc[data[fields].index_of('shared')].to_df()

    if isinstance(cross_stat, str):
        if cross_stat == 'modZscore':
            MAD_series = df.subtract(df.median(axis=1), axis=0).abs().median(axis=1)
            diff_scores = ((0.6745 * (df.subtract(df.median(axis=1), axis=0))).divide(MAD_series, axis=0)).abs()
        elif cross_stat == 'Zscore':
            diff_scores = (df.subtract(df.mean(axis=1), axis=0)).divide(df.std(axis=1), axis=0).abs()
        else:
            raise ValueError(cross_stat)
    else:
        try:
            stat = getattr(df, cross_stat.__name__)(axis=1)
        except AttributeError:
            stat = df.aggregate(cross_stat, axis=1)
        diff_scores = df.subtract(stat, axis=0).abs()

    mask = diff_scores > thresh
    for var in fields:
        flagger = flagger.setFlags(var, mask[var], **kwargs)

    return data, flagger
