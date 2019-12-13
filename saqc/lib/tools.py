#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
from typing import Sequence, Union

import numpy as np
import pandas as pd
import numba as nb

from saqc.lib.types import T

STRING_2_FUNC = {
    'sum':      np.sum,
    'mean':     np.mean,
    'median':   np.median,
    'min':      np.min,
    'max':      np.max,
    'first':    pd.Series(np.nan, index=pd.DatetimeIndex([])).resample('0min').first,
    'last':     pd.Series(np.nan, index=pd.DatetimeIndex([])).resample('0min').last
}

def assertScalar(name, value, optional=False):
    if (not np.isscalar(value)) and (value is not None) and (optional is True):
        raise ValueError(f"'{name}' needs to be a scalar or 'None'")
    elif (not np.isscalar(value)) and optional is False:
        raise ValueError(f"'{name}' needs to be a scalar")


def toSequence(
    value: Union[T, Sequence[T]], default: Union[T, Sequence[T]] = None
) -> Sequence[T]:
    if value is None:
        value = default
    if np.isscalar(value):
        value = [value]
    return value


@nb.jit(nopython=True, cache=True)
def findIndex(iterable, value, start):
    i = start
    while i < len(iterable):
        v = iterable[i]
        if v >= value:
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
        raise ValueError("strictly monotonic index needed")

    window_size = pd.to_timedelta(window_size).to_timedelta64().astype(np.int64)
    if iter_delta:
        iter_delta = pd.to_timedelta(iter_delta).to_timedelta64().astype(np.int64)

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
            start_idx = findIndex(dates, start_date + iter_delta, start_idx)
        else:
            start_idx += 1

        start_date = dates[start_idx]


def inferFrequency(data):
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))



def retrieveTrustworthyOriginal(data, field, flagger=None, level=None):
    """Columns of data passed to the saqc runner may not be sampled to its original sampling rate - thus
    differenciating between missng value - nans und fillvalue nans is impossible.

    This function:
    (1) if flagger is None:
        (a) estimates the sampling rate of the input dataseries by dropping all nans and then returns the series at the
            estimated samplng rate.

    (2) if "flagger" is not None but "level" is None:
        (a) all values are dropped, that are flagged worse then flagger.GOOD. (so unflagged values wont be dropped)
        (b) estimates the sampling rate of the input dataseries by dropping all nans and then returns the series at the
            estimated samplng rate.
    (3) if "flagger" is not None and "level" is not None:
        (a) all values are dropped, that are flagged worse then level. (so unflagged values wont be dropped)
        (b) estimates the sampling rate of the input dataseries by dropping all nans and then returns the series at the
            estimated samplng rate.

    Note, that the passed dataseries should be harmonized to an equidistant
        frequencie grid (maybe including blow up entries).

    :param data:        DataFrame. The Data frame holding the data containing 'field'.
    :param field:       String. Fieldname of the column in data, that you want to sample to original sampling rate.
                        It has to have a harmonic
    :param flagger:     None or a flagger object.
    :param level:       Lower bound of flags that are excepted for data. Must be a flag the flagger can handle.

    """
    dataseries = data[field]

    if flagger is not None:
        if level is not None:
            data_use = flagger.isFlagged(field, flag=level, comparator="<=")
        else:
            data_use = flagger.isFlagged(field, flag=flagger.GOOD, comparator="<=")
        # drop all flags that are suspicious or worse
        dataseries = dataseries[data_use]

    # drop the nan values that may result from any preceeding upsampling of the measurements:
    dataseries = dataseries.dropna()

    if dataseries.empty:
        return dataseries, np.nan

    # estimate original data sampling frequencie
    # (the original series sampling rate may not match data-input sample rate):
    seconds_rate = (dataseries.index - dataseries.index.shift(-1)).to_series().min().seconds
    data_rate = pd.tseries.frequencies.to_offset(str(seconds_rate) + 's')

    return dataseries.asfreq(data_rate), data_rate


def offset2seconds(offset):
    """Function returns total seconds upon "offset like input

    :param offset:  offset string or pandas offset object.
    """

    return pd.Timedelta.total_seconds(pd.Timedelta(offset))


def flagWindow(
    flagger_old, flagger_new, field, direction="fw", window=0, **kwargs
) -> pd.Series:

    if window == 0 or window == "":
        return flagger_new

    fw, bw = False, False
    mask = flagger_old.getFlags(field) != flagger_new.getFlags(field)
    f = flagger_new.isFlagged(field) & mask

    if not mask.any():
        # nothing was flagged, so nothing need to be flagged additional
        return flagger_new

    if isinstance(window, int):
        x = f.rolling(window=window + 1).sum()
        if direction in ["fw", "both"]:
            fw = x.fillna(method="bfill").astype(bool)
        if direction in ["bw", "both"]:
            bw = x.shift(-window).fillna(method="bfill").astype(bool)
    else:
        # time-based windows
        if direction in ["bw", "both"]:
            # todo: implement time-based backward rolling
            raise NotImplementedError
        fw = f.rolling(window=window, closed="both").sum().astype(bool)

    fmask = bw | fw
    return flagger_new.setFlags(field, fmask, **kwargs)


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
        day1 = 31 if month1 in [1, 3, 5, 7, 8, 10, 12] else 29 if month1 == 2 else 30

    # test plausibility of date
    try:
        f = "%Y-%m-%d"
        t0 = pd.to_datetime(f"2001-{month0}-{day0}", format=f)
        t1 = pd.to_datetime(f"2001-{month1}-{day1}", format=f)
    except ValueError:
        raise ValueError("Given datelike parameter not logical")

    # swap
    if t1 < t0:
        # we create the same mask as we would do if not inverted
        # but the borders need special treatment..
        # ===end]....................[start====
        # ======]end+1........start-1[=========
        # ......[end+1========start-1]......... + invert
        # ......[start`========= end`]......... + invert
        t0 -= pd.to_timedelta("1d")
        t1 += pd.to_timedelta("1d")
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


def assertDataFrame(df, argname="arg", allow_multiindex=True):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{argname} must be of type pd.DataFrame, {type(df)} was given")
    if not allow_multiindex:
        assertSingleColumns(df, argname)
    if not df.columns.is_unique:
        raise TypeError(f"{argname} must have unique columns")


def assertSeries(df, argname="arg"):
    if not isinstance(df, pd.Series):
        raise TypeError(f"{argname} must be of type pd.Series, {type(df)} was given")


def assertPandas(pdlike, argname="arg", allow_multiindex=True):
    if not isinstance(pdlike, pd.Series) and not isinstance(pdlike, pd.DataFrame):
        raise TypeError(
            f"{argname} must be of type pd.DataFrame or pd.Series, {type(pdlike)} was given"
        )
    if not allow_multiindex:
        assertSingleColumns(pdlike, argname)


def assertMultiColumns(dfmi, argname=""):
    assertDataFrame(dfmi, argname, allow_multiindex=True)
    if not isinstance(dfmi.columns, pd.MultiIndex):
        raise TypeError(
            f"given pd.DataFrame ({argname}) need to have a muliindex on columns, "
            f"instead it has a {type(dfmi.columns)}"
        )


def assertSingleColumns(df, argname=""):
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        raise TypeError(
            f"given pd.DataFrame {argname} is not allowed to have a muliindex on columns"
        )

def funcInput_2_func(func):
    """
    Aggregation functions passed by the user, are selected by looking them up in the STRING_2_DICT dictionary -
    But since there are wrappers, that dynamically generate aggregation functions and pass those on ,the parameter
    interfaces must as well be capable of processing real functions passed. This function does that.

    :param func: A key to the STRING_2_FUNC dict, or an actual function
    """
    # if input is a callable - than just pass it:
    if hasattr(func, "__call__"):
        return func
    elif func in STRING_2_FUNC.keys():
        return STRING_2_FUNC[func]
    else:
        raise ValueError("Function input not a callable nor a known key to internal the func dictionary.")
