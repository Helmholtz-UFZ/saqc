#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Sequence, Union, Any, Iterator

import numpy as np
import numba as nb
import pandas as pd
import logging
import dios


# from saqc.flagger import BaseFlagger
from saqc.lib.types import T

logger = logging.getLogger("SaQC")

def assertScalar(name, value, optional=False):
    if (not np.isscalar(value)) and (value is not None) and (optional is True):
        raise ValueError(f"'{name}' needs to be a scalar or 'None'")
    elif (not np.isscalar(value)) and optional is False:
        raise ValueError(f"'{name}' needs to be a scalar")


def toSequence(value: Union[T, Sequence[T]], default: Union[T, Sequence[T]] = None) -> Sequence[T]:
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
    """
    this function is a building block of a custom implementation of
    the pandas 'rolling' method. A number of shortcomings in the
    'rolling' implementation might made this a worthwhil endavour,
    namly:
    + There is no way to provide a step size, i.e. to not start the
      next rolling window at the very next row in the DataFrame/Series
    + The inconsistent bahaviour with numerical vs frequency based
      window sizes. When winsz is an integer, all windows are equally
      large (winsz=5 -> windows contain 5 elements), but variable in
      size, when the winsz is a frequency string (winsz="2D" ->
      window grows from size 1 during the first iteration until it
      covers the given frequency). Especially the bahaviour with
      frequency strings is quite unfortunate when calling methods
      relying on the size of the window (sum, mean, median)
    """

    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("Must pass pd.DatetimeIndex")

    # lets work on numpy data structures for performance reasons
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


def inferFrequency(data: pd.Series) -> pd.DateOffset:
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def retrieveTrustworthyOriginal(
    data: dios.DictOfSeries, field: str, flagger=None, level: Any = None
) -> dios.DictOfSeries:
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
        mask = flagger.isFlagged(field, flag=level or flagger.GOOD, comparator="<=")
        # drop all flags that are suspicious or worse
        dataseries = dataseries[mask]

    # drop the nan values that may result from any preceeding upsampling of the measurements:
    dataseries = dataseries.dropna()

    if dataseries.empty:
        return dataseries, np.nan

    # estimate original data sampling frequencie
    # (the original series sampling rate may not match data-input sample rate):
    seconds_rate = dataseries.index.to_series().diff().min().seconds
    data_rate = pd.tseries.frequencies.to_offset(str(seconds_rate) + "s")

    return dataseries.asfreq(data_rate), data_rate


def offset2seconds(offset):
    """Function returns total seconds upon "offset like input

    :param offset:  offset string or pandas offset object.
    """

    return pd.Timedelta.total_seconds(pd.Timedelta(offset))


def flagWindow(flagger_old, flagger_new, field, direction="fw", window=0, **kwargs) -> pd.Series:
    # NOTE: unused -> remove?
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


def assertDictOfSeries(df: Any, argname: str = "arg") -> None:
    if not isinstance(df, dios.DictOfSeries):
        raise TypeError(f"{argname} must be of type dios.DictOfSeries, {type(df)} was given")


def assertSeries(srs: Any, argname: str = "arg") -> None:
    if not isinstance(srs, pd.Series):
        raise TypeError(f"{argname} must be of type pd.Series, {type(srs)} was given")


@nb.jit(nopython=True, cache=True)
def otherIndex(values: np.ndarray, start: int = 0) -> int:
    """
    returns the index of the first non value not equal to values[0]
    -> values[start:i] are all identical
    """
    val = values[start]
    for i in range(start, len(values)):
        if values[i] != val:
            return i
    return -1


def groupConsecutives(series: pd.Series) -> Iterator[pd.Series]:
    """
    group consecutive values into distinct pd.Series
    """
    index = series.index
    values = series.values

    start = 0
    while True:
        stop = otherIndex(values, start)
        if stop == -1:
            break
        yield pd.Series(data=values[start:stop], index=index[start:stop])
        start = stop


def mergeDios(left, right, join="merge"):
    # use dios.merge() as soon as it implemented
    # see https://git.ufz.de/rdm/dios/issues/15

    merged = left.copy()
    shared_cols = left.columns.intersection(right.columns)
    for c in shared_cols:
        l, r = left[c], right[c]
        if join == "merge":
            # NOTE:
            # our merge behavior is nothing more than an
            # outer join, where the right join argument
            # overwrites the left at the shared indices,
            # while on a normal outer join common indices
            # hold the values from the left join argument
            r, l = l.align(r, join="outer")
        else:
            l, r = l.align(r, join=join)
        merged[c] = l.combine_first(r)

    newcols = right.columns.difference(merged.columns)
    for c in newcols:
        merged[c] = right[c].copy()

    return merged


def isQuoted(string):
    return bool(re.search(r"'.*'|\".*\"", string))


def dropper(field, drop_flags, flagger, default):
    drop_mask = pd.Series(False, flagger.getFlags(field).index)
    if drop_flags is None:
        drop_flags = default
    drop_flags = toSequence(drop_flags)
    if len(drop_flags) > 0:
        drop_mask |= flagger.isFlagged(field, flag=drop_flags)
    return drop_mask


def mutateIndex(index, old_name, new_name):
    pos = index.get_loc(old_name)
    index = index.drop(index[pos])
    index = index.insert(pos, new_name)
    return index


def _sampling_mode_iterator(sub_index_dict, uniformity_dict, sample_rate_dict, x_data, bin_accuracy=60,
                            min_bandwidth_share=0.1):
    """
    the function is called by the "estimate_sampling_rates" function.

    Its purpose is to decompose a given index into its different sampling frequencies and return
    frequencies and indices belonging to a frequencies sampling.

    The "bin_accuracy" parameter refers to the detection accuracy. It has dimension of seconds.

    The "min_bandwidth_share" refers to the minimum percentage the values associated with a frequencie must contribute
    to the total number of samples, to be considered a significant frequency mode of the index.
    (0.1 means, you can have up to 10 different frequencies, consisting of 10 percent of the total values each.)
    """


    out_sub_dict = sub_index_dict.copy()
    out_uni_dict = uniformity_dict.copy()
    out_rate_dict = sample_rate_dict.copy()
    for mode in sub_index_dict.keys():
        if not uniformity_dict[mode]:
            x_data_diff = np.diff(x_data[sub_index_dict[mode]])
            q_mask = np.logical_and(np.quantile(x_data_diff, 0.01) - 60 < x_data_diff,
                                x_data_diff < np.quantile(x_data_diff, 0.99) + 60)
            x_cutted_of = x_data_diff[q_mask]
            bins = np.arange(30, int(np.ceil(max(x_cutted_of))) + 90)[::bin_accuracy]
            bins = np.concatenate((np.array([0]), bins))
            hist, bins = np.histogram(x_cutted_of, bins=bins)
            sub_modes = np.where(hist > len(x_data) / min_bandwidth_share)[0]
            if len(sub_modes) == 1:
                out_uni_dict[mode] = True
                out_rate_dict[mode] = (bins[sub_modes[0]], bins[sub_modes[0] + 1])
            elif len(sub_modes) > 1:
                sub_count = 1
                for sub_mode in sub_modes:
                    sub_index = np.where(np.logical_and(bins[sub_mode] < x_data_diff,
                                                        x_data_diff < bins[sub_mode + 1]))[0]
                    new_mode_name = mode + '.' + str(sub_count)
                    out_sub_dict[new_mode_name] = sub_index
                    out_uni_dict[new_mode_name] = False
                    sub_count += 1
                out_sub_dict.pop(mode)
                out_uni_dict.pop(mode)
    return out_sub_dict, out_uni_dict, out_rate_dict


def estimate_sampling_rates(index, freq=None):
    """
    Function estimates the sampling rate(s) an index includes.
    If freq is passed, additionally a warning is logged, if freq is inconsistent with the sampling rate estimate.

    In the current implementation, estimation accuracy is one Minute. (With an extra bin for frequencies < 30 seconds)
    So the functions purpose is not to detect slight drifts in the frequencie, but to detect mixing of/changing between
    significantly differing sampling rates.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index, the sampling modes are estimated of.
    freq : Offsetstring or None, default None
        Frequencie of wich consistence with the estimate is checked. None (default) skips check.

    Returns
    -------
    sample_rates : set
        Set of Tuples (x,y). Any tuple indicates that tthere is a sampling frequency f in the index detectable,
        so that "x seconds" < f.seconds < "y seconds".

    """
    index_data = index.to_numpy(float)
    x_data = index_data * 10 ** (-9)
    sub_index_dict = {'mode_1': np.arange(0, len(x_data))}
    uniformity_dict = {'mode_1': False}
    sample_rate_dict = {}
    k = 0
    while any(val == False for val in uniformity_dict.values()):
        sub_index_dict, uniformity_dict, sample_rate_dict = _sampling_mode_iterator(sub_index_dict, uniformity_dict,
                                                                               sample_rate_dict, x_data)
        if k > 20:
            logger.warning('Sample rate estimation failed. Too many iterations while splitting into modes.')
            break
    sample_rates = set(sample_rate_dict.values())
    if len(sample_rates) > 1:
        logger.warning('Multiple sampling modes detected: {}'.format(str(sample_rates)
                                                                      + ' (min seconds, max seconds)'))
    if freq:
        t_seconds = pd.Timedelta(freq).total_seconds()
        eval_freq = any([True if x < t_seconds < y else False for (x, y) in sample_rates])
        if not eval_freq:
            logger.warning('Frequency passed does not fit any of the estimated data sampling modes.')

    return sample_rates