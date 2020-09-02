#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Sequence, Union, Any, Iterator

import numpy as np
import numba as nb
import pandas as pd
from scipy import fft
import logging

import dios
import inspect

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


def dropper(field, to_drop, flagger, default):
    drop_mask = pd.Series(False, flagger.getFlags(field).index)
    if to_drop is None:
        to_drop = default
    to_drop = toSequence(to_drop)
    if len(to_drop) > 0:
        drop_mask |= flagger.isFlagged(field, flag=to_drop)
    return drop_mask


def mutateIndex(index, old_name, new_name):
    pos = index.get_loc(old_name)
    index = index.drop(index[pos])
    index = index.insert(pos, new_name)
    return index


def estimateFrequency(index, delta_precision=-1, max_rate="10s", min_rate="1D", pad_fft_to_opt=True,
                      min_energy=0.2, max_freqs=10, bins=None):

    """
    Function to estimate the sampling rate of an index.

    The function comes with some optional overhead.
    The default options detect sampling rates from 10 seconds to 1 day with a 10 seconds precision
    for sampling rates below one minute, and a one minute precision for rates between 1 minute and
    one day.

    The function is designed to detect mixed sampling ratess as well as rate changes.
    In boh situations, all the sampling rates detected, are returned, together with their
    greatest common rate.

    Parameters
    ----------
    index : {pandas.DateTimeIndex}
        The index of wich the sampling rates shall be estimated
    delta_precision : int, default -1
        determines detection precision. Precision equals: seconds*10**(-1-delta_precision).
        A too high precision attempt can lead to performance loss and doesnt necessarily result in
        a more precise result. Especially when the samples deviation from their mean rate
        is high compared to the delta_precision.
    max_rate : str, default "10s"
        Maximum rate that can be detected.
    min_rate : str, default "1D"
        Minimum detectable sampling rate.
    pad_fft_to_opt : bool, default True
        Wheather or not to speed up fft application by zero padding the derived response series to
        an optimal length. (Length = 2**N)
    min_energy : float, default 0.2
        min_energy : percentage of energy a sampling rate must represent at least, to be detected. Lower values
        result in higher sensibillity - but as well increas detection rate of mix products. Default proofed to be
        stable.
    max_freqs : int, default 10
        Maximum number of frequencies collected from the index. Mainly a value to prevent the frequency
        collection loop from collecting noise and running endlessly.
    bins : {None, List[float]} : default None

    Returns
    -------
        freq : {None, str}
            Either the sampling rate that was detected in the sample index (if uniform). Or
            the greates common rate of all the sampling rates detected. Equals `None` if
            detection failed and `"empty"`, if input index was empty.
        freqs : List[str]
            List of detected sampling rates.

    """
    index_n = index.to_numpy(float)
    if index.empty:
        return 'empty', []

    index_n = (index_n - index_n[0])*10**(-9 + delta_precision)
    delta = np.zeros(int(index_n[-1])+1)
    delta[index_n.astype(int)] = 1
    if pad_fft_to_opt:
        delta_f = np.abs(fft.rfft(delta, fft.next_fast_len(len(delta))))
    else:
        delta_f = np.abs(fft.rfft(delta))

    len_f = len(delta_f)*2
    min_energy = delta_f[0]*min_energy
    # calc/assign low/high freq cut offs (makes life easier):
    min_rate_i = int(len_f/(pd.Timedelta(min_rate).total_seconds()*(10**delta_precision)))
    delta_f[:min_rate_i] = 0
    max_rate_i = int(len_f/(pd.Timedelta(max_rate).total_seconds()*(10**delta_precision)))
    hf_cutoff = min(max_rate_i, len_f//2)
    delta_f[hf_cutoff:] = 0
    delta_f[delta_f < min_energy] = 0

    # find frequencies present:
    freqs = []
    f_i = np.argmax(delta_f)
    while (f_i > 0) & (len(freqs) < max_freqs):
        f = (len_f / f_i)/(60*10**(delta_precision))
        freqs.append(f)
        for i in range(1, hf_cutoff//f_i + 1):
            delta_f[(i*f_i) - min_rate_i:(i*f_i) + min_rate_i] = 0
        f_i = np.argmax(delta_f)

    if len(freqs) == 0:
        return None, []

    if bins is None:
        r = range(0, int(pd.Timedelta(min_rate).total_seconds()/60))
        bins = [0, 0.1, 0.2, 0.3, 0.4] + [i + 0.5 for i in r]

    f_hist, bins = np.histogram(freqs, bins=bins)
    freqs = np.ceil(bins[:-1][f_hist >= 1])
    gcd_freq = np.gcd.reduce((10*freqs).astype(int))/10

    return str(int(gcd_freq)) + 'min', [str(int(i)) + 'min' for i in freqs]


def evalFreqStr(freq, index):
    if freq == 'auto':
        freq = index.inferred_freq
        if freq is None:
            freq, freqs = estimateFrequency(index)
        if freq is None:
            raise TypeError('Sampling rate could not be estimated, nor was an explicit freq string '
                            'delivered.')
        if len(freqs) > 1:
            logging.warning(f"Sampling rate not uniform!."
                            f"Detected: {freqs}"
                            f"Greatest common Rate: {freq}, got selected.")
    return freq
