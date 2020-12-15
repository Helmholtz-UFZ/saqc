#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Sequence, Union, Any, Iterator

import itertools
import numpy as np
import numba as nb
import pandas as pd
from scipy import fft
import logging
import dios

import collections
from scipy.cluster.hierarchy import linkage, fcluster
from saqc.lib.types import T

# keep this for external imports
from saqc.lib.rolling import customRoller

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


def seasonalMask(dtindex, season_start, season_end, include_bounds):
    """
    This function generates date-periodic/seasonal masks from an index passed.

    For example you could mask all the values of an index, that are sampled in winter, or between 6 and 9 o'clock.
    See the examples section for more details.

    Parameters
    ----------
    dtindex : pandas.DatetimeIndex
        The index according to wich you want to generate a mask.
        (=resulting mask will be indexed with 'dtindex')
    season_start : str
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `season_end` parameter.
        See examples section below for some examples.
    season_end : str
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `season_end` parameter.
        See examples section below for some examples.
    include_bounds : boolean
        Wheather or not to include the mask defining bounds to the mask.

    Returns
    -------
    to_mask : pandas.Series[bool]
        A series, indexed with the input index and having value `True` for all the values that are to be masked.

    Examples
    --------
    The `season_start` and `season_end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `season_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:

    >>> season_start = "01T15:00:00"
    >>> season_end = "13T17:30:00"

    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked

    >>> season_start = "01:00"
    >>> season_end = "04:00"

    All the values between the first and 4th minute of every hour get masked.

    >>> season_start = "01-01T00:00:00"
    >>> season_end = "01-03T00:00:00"

    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:

    >>> season_start = "01-01T00:00:00"
    >>> season_end = "02-28T23:59:59"

    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:

    >>> season_start = "22:00:00"
    >>> season_end = "06:00:00"

    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    def _replaceBuilder(stamp):
        keys = ("second", "minute", "hour", "day", "month", "year")
        stamp_list = map(int, re.split(r"[-T:]", stamp)[::-1])
        stamp_kwargs = dict(zip(keys, stamp_list))

        def _replace(index):
            if "day" in stamp_kwargs:
                stamp_kwargs["day"] = min(stamp_kwargs["day"], index[0].daysinmonth)

            out = index[0].replace(**stamp_kwargs)
            return out.strftime("%Y-%m-%dT%H:%M:%S")

        return _replace

    mask = pd.Series(include_bounds, index=dtindex)

    start_replacer = _replaceBuilder(season_start)
    end_replacer = _replaceBuilder(season_end)

    if pd.Timestamp(start_replacer(dtindex)) <= pd.Timestamp(end_replacer(dtindex)):
        def _selector(x, base_bool=include_bounds):
            x[start_replacer(x.index):end_replacer(x.index)] = not base_bool
            return x
    else:
        def _selector(x, base_bool=include_bounds):
            x[:end_replacer(x.index)] = not base_bool
            x[start_replacer(x.index):] = not base_bool
            return x

    freq = '1' + 'mmmhhhdddMMMYYY'[len(season_start)]
    return mask.groupby(pd.Grouper(freq=freq)).transform(_selector)


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


def mergeDios(left, right, subset=None, join="merge"):
    # use dios.merge() as soon as it implemented
    # see https://git.ufz.de/rdm/dios/issues/15

    merged = left.copy()
    if subset is not None:
        right_subset_cols = right.columns.intersection(subset)
    else:
        right_subset_cols = right.columns

    shared_cols = left.columns.intersection(right_subset_cols)

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

    newcols = right_subset_cols.difference(left.columns)
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


def estimateFrequency(index, delta_precision=-1, max_rate="10s", min_rate="1D", optimize=True,
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

    Note, that there is a certain lower bound of index length,
    beneath which frequency leaking and Niquist limit take over and mess up the fourier
    transform.

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
    optimize : bool, default True
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
    if optimize:
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


def evalFreqStr(freq, check, index):
    if check in ['check', 'auto']:
        f_passed = freq
        freq = index.inferred_freq
        freqs = [freq]
        if freq is None:
            freq, freqs = estimateFrequency(index)
        if freq is None:
            logging.warning('Sampling rate could not be estimated.')
        if len(freqs) > 1:
            logging.warning(f"Sampling rate seems to be not uniform!."
                            f"Detected: {freqs}")

        if check == 'check':
            f_passed_seconds = pd.Timedelta(f_passed).total_seconds()
            freq_seconds = pd.Timedelta(freq).total_seconds()
            if (f_passed_seconds != freq_seconds):
                logging.warning(f"Sampling rate estimate ({freq}) missmatches passed frequency ({f_passed}).")
        elif check == 'auto':
            if freq is None:
                raise ValueError('Frequency estimation for non-empty series failed with no fall back frequency passed.')
            f_passed = freq
    else:
        f_passed = freq
    return f_passed


def detectDeviants(data, metric, norm_spread, norm_frac, linkage_method='single', population='variables'):
    """
    Helper function for carrying out the repeatedly upcoming task,
    of detecting variables a group of variables.

    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed
    in respect to a certain metric and linkage method.

    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".

    Note, that the function also can be used to detect anormal regimes in a variable by assigning the different regimes
    dios.DictOfSeries columns and passing this dios.

    Parameters
    ----------
    data : {pandas.DataFrame, dios.DictOfSeries}
        Input data
    metric : Callable[[numpy.array, numpy.array], float]
        A metric function that for calculating the dissimilarity between 2 variables.
    norm_spread : float
        A threshold denoting the distance, members of the "normal" group must not exceed to each other (in terms of the
        metric passed) to qualify their group as the "normal" group.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables or samples,
        the "normal" group has to comprise to be the normal group actually.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the variables.
    population : {"variables", "samples"}
        Wheather to relate the minimum percentage of values needed to form a normal group, to the total number of
        variables or to the total number of samples.

    Returns
    -------
    deviants : List
        A list containing the column positions of deviant variables in the input frame/dios.

    """
    var_num = len(data.columns)
    if var_num <= 1:
        return []
    dist_mat = np.zeros((var_num, var_num))
    combs = list(itertools.combinations(range(0, var_num), 2))
    for i, j in combs:
        dist = metric(data.iloc[:, i].values, data.iloc[:, j].values)
        dist_mat[i, j] = dist

    condensed = np.abs(dist_mat[tuple(zip(*combs))])
    Z = linkage(condensed, method=linkage_method)
    cluster = fcluster(Z, norm_spread, criterion='distance')
    if population == 'variables':
        counts = collections.Counter(cluster)
        pop_num = var_num
    elif population == 'samples':
        counts = {cluster[j]: 0 for j in range(0,var_num)}
        for c in range(var_num):
            counts[cluster[c]] += data.iloc[:, c].dropna().shape[0]
        pop_num = np.sum(list(counts.values()))
    else:
        raise ValueError("Not a valid normality criteria keyword passed. Pass either 'variables' or 'population'.")
    norm_cluster = -1

    for item in counts.items():
        if item[1] > norm_frac * pop_num:
            norm_cluster = item[0]
            break

    if norm_cluster == -1 or counts[norm_cluster] == pop_num:
        return []
    else:
        return [i for i, x in enumerate(cluster) if x != norm_cluster]


