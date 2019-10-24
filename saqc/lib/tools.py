#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
from typing import Union

import numpy as np
import pandas as pd
import numba as nb
import logging
import sys

from ..lib.types import PandasLike, ArrayLike


@nb.jit(nopython=True, cache=True)
def findIndex(iterable, value, start):
    i = start
    while i < len(iterable):
        v = iterable[i]
        if v >= value:
            # if v == value:
                # include the end_date if present
                # return i + 1
            return i
        i = i + 1
    return -1


@nb.jit(nopython=True, cache=True)
def valueRange(iterable):
    minval = iterable[0]
    maxval = minval
    for v in iterable:
        if v < minval:
            minval = v
        elif v > maxval:
            maxval = v
    return maxval - minval


def slidingWindowIndices(dates, window_size, iter_delta=None):

    # lets work on numpy data structures for performance reasons
    if isinstance(dates, pd.DataFrame):
        dates = dates.index
    dates = np.array(dates, dtype=np.int64)
    if np.any(np.diff(dates) <= 0):
        raise ValueError("strictly monotic index needed")

    window_size = pd.to_timedelta(window_size, box=False).astype(np.int64)
    if iter_delta:
        iter_delta = pd.to_timedelta(iter_delta, box=False).astype(np.int64)

    start_date = dates[0]
    last_date = dates[-1]
    start_idx = 0
    end_idx = start_idx

    while True:
        end_date = start_date + window_size
        if (end_date > last_date) or (start_idx == -1) or (end_idx == -1):
            break

        end_idx = findIndex(dates, end_date, end_idx)
        yield start_idx, end_idx

        if iter_delta:
            start_idx = findIndex(dates, start_date+iter_delta, start_idx)
        else:
            start_idx += 1

        start_date = dates[start_idx]


def numpyfy(arg:  Union[PandasLike,
                        np.ndarray,
                        numbers.Number]) -> np.ndarray:
    try:
        # pandas dataframe
        return arg.values
    except AttributeError:
        try:
            # numpy array
            return arg.copy()
        except AttributeError:
            # scalar
            return np.atleast_1d(arg)


def broadcastMany(*args: ArrayLike) -> np.ndarray:
    arrays = [np.atleast_1d(a) for a in args]
    target_ndim = max(arr.ndim for arr in arrays)
    out = []
    for arr in arrays:
        out.append(arr[(slice(None),) + (None,) * (target_ndim - arr.ndim)])
    target_shape = np.broadcast(*out).shape
    return tuple(np.broadcast_to(arr, target_shape) for arr in out)


def inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def estimateSamplingRate(index):
    """The function estimates the sampling rate of a datetime index.
    The estimation basically evaluates a histogram of bins with seconds-accuracy. This means, that the
    result may be contra intuitive or trashy very likely, if the input series is not rastered (harmonized with skips)
    to an interval divisible by seconds.

    :param index: A DatetimeIndex or array like Datetime listing, of wich you want the sampling rate to be
                  estimated.
    """

    if index.empty:
        return pd.tseries.frequencies.to_offset('0s')
    scnds_series = (pd.Series(index).diff().dt.total_seconds()).dropna()
    max_scnds = scnds_series.max()
    min_scnds = scnds_series.min()
    hist = np.histogram(scnds_series, range=(min_scnds, max_scnds + 1), bins=int(max_scnds - min_scnds + 1))
    # return smallest non zero sample difference (this works, because input is expected to be at least
    # harmonized with skips)
    return pd.tseries.frequencies.to_offset(str(int(hist[1][:-1][hist[0] > 0].min())) + 's')


def retrieveTrustworthyOriginal(dataseries, dataflags=None, flagger=None):
    """Columns of data passed to the saqc runner may not be sampled to its original sampling rate - thus
    differenciating between missng value - nans und fillvalue nans is impossible. This function evaluates flags for a
    passed series, if flags and flagger object are passed and downsamples the input series to its original sampling
    rate and sparsity.

    :param dataseries:  The pd.dataseries object that you want to sample to original rate. It has to have a harmonic
                        timestamp.
    :param dataflags:   the flags series,referring to the passed dataseries.
    :param dataflags:   A flagger object, to apply the passed flags onto the dataseries.

    """
    if (dataflags is not None) and (flagger is not None):
        data_use = flagger.isFlagged(dataflags, flag=flagger.flags.good(), comparator='==') | \
                   flagger.isFlagged(dataflags, flag=flagger.flags.unflagged(), comparator='==')
        # drop suspicious values
        dataseries = dataseries[data_use.values]
    # additionally, drop the nan values that result from any preceeding upsampling of the
    # measurements:
    dataseries = dataseries.dropna()
    # eventually, after dropping all nans, there is nothing left:
    if dataseries.empty:
        return dataseries, np.nan
    # estimate original data sampling frequencie (the original series sampling rate may not match data-input sample
    # rate):
    data_rate = estimateSamplingRate(dataseries.index)
    # resample dataseries to its original sampling rate (now certain, to only get nans, indeed denoting "missing" data)
    return dataseries.resample(data_rate).asfreq(), data_rate


def offset2seconds(offset):
    """Function returns total seconds upon "offset like input

    :param offset:  offset string or pandas offset object.
    """

    return pd.Timedelta.total_seconds(pd.Timedelta(offset))


def offset2periods(input_offset, period_offset):
    """Function returns the number of periods of length "periods_offset" that sum up to length "input offset".
    (Namely their fraction.)

    :param input_offset         Offset string or pandas offset object.
    :param period_offset        Offset string or pandas offset object.
    """

    return offset2seconds(input_offset) / offset2seconds(period_offset)


def getPandasVarNames(pandas_like):
    """The function is a workaround, to not have to implement case distinctions whenever your not sure if your
    asking for a dataframes columns or a series name (=1_D Dataframe)' """
    if isinstance(pandas_like, pd.DataFrame):
        return pandas_like.columns
    if isinstance(pandas_like, pd.Series):
        return [pandas_like.name]


def getPandasData(pandas_like, fieldname):
    """The function is a workaround, to not have to implement case distinctions whenever your not sure if your
       asking for a dataframes columns data or a series.
       Note, that indexing by integer is possible as well. With fieldname=0, you will get a series passed, if the input
       was a series, and the first dataframes column, if input was a DataFrame"""

    # DF input
    if isinstance(pandas_like, pd.DataFrame):
        if isinstance(fieldname, int):
            return pandas_like.iloc[:, fieldname]
        if isinstance(fieldname, str):
            return pandas_like[fieldname]

    # Series input
    if isinstance(pandas_like, pd.Series):
        return pandas_like


def checkQCParameters(para_dict, called_by):
    """The function is designed to check parameter passed to the QC functions. The checking method is determined by
    a nested dictionary that has to be passed to the parameter para dict and should have the following form:

    check_dict = {value_name_1: {'value': value,
                                 'type': [class_1, class_2, ..., classM],
                                 'range': [min , max],
                                 'member':[e1, e2, ...,eN]}}
                                 'tests': {test_1_name: test_func1,
                                           test_2_name: test_func2,...
                                           test_M_name: test_funcM},
                value_name_2: {....}}
    """

    global_checker = 0

    for para in para_dict.keys():

        local_checker = 0
        sub_dict = para_dict[para]

        # check the type
        if 'type' in sub_dict.keys():
            type_check = 0
            for type in sub_dict['type']:
                if isinstance(sub_dict['value'], type):
                    type_check += 1
            if type_check == 0:
                logging.error('Parameter {} passed to Function {} didnt pass type Test. '
                              'It has to be one out of: '.format(para, called_by, str(sub_dict['type'])))
                local_checker -= 1

        if local_checker < 0:
            global_checker -= 1
            continue

        # check range
        if 'range' in sub_dict.keys():
            if sub_dict['value'] is not None:
                if not (sub_dict['range'][0] <= sub_dict['value'] <= sub_dict['range'][1]):
                    logging.error('Parameter {} passed to Function {}, didnt pass range Test. '
                                  'Range restrains for this parameter are: '
                                  '[{}, {}]: '.format(para, called_by, str(sub_dict['range'][0]), sub_dict['range'][1]))
                    local_checker -= 1

        if local_checker < 0:
            global_checker -= 1
            continue

        # member test
        if 'member' in sub_dict.keys():
            if not (sub_dict['value'] in sub_dict['member']):
                logging.error('Parameter {} passed to Function {}, didnt pass member-of Test. '
                              'The parameter has to be one out of: '
                              '{}.'.format(para, called_by, str(sub_dict['member'])))
                local_checker -= 1

        if local_checker < 0:
            global_checker -= 1
            continue

        # apply individual tests
        if 'tests' in sub_dict.keys():
            test_dict = sub_dict['tests']
            for test in test_dict.keys():
                try:
                    test_check = test_dict[test](sub_dict['value'])
                except:
                    logging.error('Parameter {}, passed to Function {}, caused the Exception: {},'
                                  ' when tested by {} - test.'.format(para, called_by, sys.exc_info()[0], test))
                    test_check = False
                if test_check is False:
                    logging.error('Parameter {}, passed to Function {}, didnt pass the {} - '
                                  'test.'.format(para, called_by, test))
                    local_checker -= 1
                    break

            if local_checker < 0:
                global_checker -= 1
                continue

    return global_checker


def sesonalMask(dtindex, month0=1, day0=1, month1=12, day1=None):
    """
    This function provide a mask for a sesonal time range in the given dtindex.
    This means the interval is applied again on every year and even over the change of a year.
    Note that both edges are inclusive.

    Examples:
        sesonal(dtindex, 1, 1, 3, 1)  -> [jan-mar]
        sesonal(dtindex, 8, 1, 8, 15) -> [1.aug-15.aug]


    This also works, if the second border is smaller then the first

    Examples:
        sesonal(dtindex, 10, 1, 2, 1) -> [1.nov-1.feb (following year)]
        sesonal(dtindex, 1, 10, 1, 1)  -> [10.jan-1.jan(following year)] like everything except ]1.jan-10.jan[

    """
    if day1 is None:
        day1 = 31 if month1 in [1,3,5,7,8,10,12] else 29 if month1 == 2 else 30

    # test plausibility of date
    try:
        f = '%Y-%m-%d'
        t0 = pd.to_datetime(f'2001-{month0}-{day0}', format=f)
        t1 = pd.to_datetime(f'2001-{month1}-{day1}', format=f)
    except ValueError:
        raise ValueError('Given datelike parameter not logical')

    # swap
    if t1 < t0:
        # we create the same mask as we would do if not inverted
        # but the borders need special treatment..
        # ===end]....................[start====
        # ======]end+1........start-1[=========
        # ......[end+1========start-1]......... + invert
        # ......[start`========= end`]......... + invert
        t0 -= pd.to_timedelta('1d')
        t1 += pd.to_timedelta('1d')
        invert = True
        # only swap id condition is still true
        t0, t1 = t1, t0 if t1 < t0 else (t0, t1)

        month0, day0 = t0.month, t0.day
        month1, day1 = t1.month, t1.day
    else:
        invert = False

    month = [m for m in range(month0, month1 + 1)]

    # make a mask for [start:end]
    mask = dtindex.month.isin(month)
    if day0 > 1:
        exclude = [d for d in range(1, day0)]
        mask &= ~(dtindex.month.isin([month0]) & dtindex.day.isin(exclude))
    if day1 < 31:
        exclude = [d for d in range(day1 + 1, 31 + 1)]
        mask &= ~(dtindex.month.isin([month1]) & dtindex.day.isin(exclude))

    if invert:
        return ~mask
    else:
        return mask


def setup():
    pd.set_option('mode.chained_assignment', 'warn')
