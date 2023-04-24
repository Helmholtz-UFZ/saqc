#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import collections
import functools
import itertools
import re
import warnings
from typing import Any, Callable, Collection, List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from scipy import fft
from scipy.cluster.hierarchy import fcluster, linkage

from saqc.lib.types import CompT

T = TypeVar("T", str, float, int)


def assertScalar(name, value, optional=False):
    if optional and value is None:
        return
    if np.isscalar(value):
        return

    msg = f"'{name}' needs to be a scalar"
    if optional:
        msg += " or 'None'"
    raise ValueError(msg)


def toSequence(value: T | Sequence[T]) -> List[T]:
    if value is None:  # special case
        return [None]
    if isinstance(value, T.__constraints__):
        return [value]
    return list(value)


def squeezeSequence(value: Sequence[T]) -> Union[T, Sequence[T]]:
    if len(value) == 1:
        return value[0]
    return value


def periodicMask(dtindex, season_start, season_end, include_bounds):
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
    dfilter : pandas.Series[bool]
        A series, indexed with the input index and having value `False` for all the values that are to be masked.

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
            if len(index) == 0:
                return None
            if "day" in stamp_kwargs:
                stamp_kwargs["day"] = min(stamp_kwargs["day"], index[0].daysinmonth)
            out = index[0].replace(**stamp_kwargs)
            return out.strftime("%Y-%m-%dT%H:%M:%S")

        return _replace

    mask = pd.Series(True, index=dtindex)

    start_replacer = _replaceBuilder(season_start)
    end_replacer = _replaceBuilder(season_end)

    invert = False

    if pd.Timestamp(start_replacer(dtindex)) > pd.Timestamp(end_replacer(dtindex)):
        start_replacer, end_replacer = end_replacer, start_replacer
        include_bounds = not include_bounds
        invert = True

    if include_bounds:

        def _selector(x):
            x[start_replacer(x.index) : end_replacer(x.index)] = False
            return x

    else:

        def _selector(x):
            s = start_replacer(x.index)
            e = end_replacer(x.index)
            x[s:e] = False
            x[s:s] = True
            x[e:e] = True
            return x

    freq = "1" + "mmmhhhdddMMMYYY"[len(season_start)]
    out = mask.groupby(pd.Grouper(freq=freq)).transform(_selector)
    if invert:
        out = ~out
    return out


def isQuoted(string):
    return bool(re.search(r"'.*'|\".*\"", string))


def mutateIndex(index, old_name, new_name):
    pos = index.get_loc(old_name)
    index = index.drop(index[pos])
    index = index.insert(pos, new_name)
    return index


def estimateFrequency(
    index,
    delta_precision=-1,
    max_rate="10s",
    min_rate="1D",
    optimize=True,
    min_energy=0.2,
    max_freqs=10,
    bins=None,
):
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
        return "empty", []

    index_n = (index_n - index_n[0]) * 10 ** (-9 + delta_precision)
    delta = np.zeros(int(index_n[-1]) + 1)
    delta[index_n.astype(int)] = 1
    if optimize:
        delta_f = np.abs(fft.rfft(delta, fft.next_fast_len(len(delta))))
    else:
        delta_f = np.abs(fft.rfft(delta))

    len_f = len(delta_f) * 2
    min_energy = delta_f[0] * min_energy
    # calc/assign low/high freq cut offs (makes life easier):
    min_rate_i = int(
        len_f / (pd.Timedelta(min_rate).total_seconds() * (10**delta_precision))
    )
    delta_f[:min_rate_i] = 0
    max_rate_i = int(
        len_f / (pd.Timedelta(max_rate).total_seconds() * (10**delta_precision))
    )
    hf_cutoff = min(max_rate_i, len_f // 2)
    delta_f[hf_cutoff:] = 0
    delta_f[delta_f < min_energy] = 0

    # find frequencies present:
    freqs = []
    f_i = np.argmax(delta_f)
    while (f_i > 0) & (len(freqs) < max_freqs):
        f = (len_f / f_i) / (60 * 10 ** (delta_precision))
        freqs.append(f)
        for i in range(1, hf_cutoff // f_i + 1):
            delta_f[(i * f_i) - min_rate_i : (i * f_i) + min_rate_i] = 0
        f_i = np.argmax(delta_f)

    if len(freqs) == 0:
        return None, []

    if bins is None:
        r = range(0, int(pd.Timedelta(min_rate).total_seconds() / 60))
        bins = [0, 0.1, 0.2, 0.3, 0.4] + [i + 0.5 for i in r]

    f_hist, bins = np.histogram(freqs, bins=bins)
    freqs = np.ceil(bins[:-1][f_hist >= 1])
    gcd_freq = np.gcd.reduce((10 * freqs).astype(int)) / 10

    return str(int(gcd_freq)) + "min", [str(int(i)) + "min" for i in freqs]


def detectDeviants(
    data,
    metric,
    norm_spread,
    norm_frac,
    linkage_method="single",
    population="variables",
):
    """
    Helper function for carrying out the repeatedly upcoming task,
    of detecting variables a group of variables.

    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed
    in respect to a certain metric and linkage method.

    In addition, only a group is considered "normal" if it contains more then `frac` percent of the
    variables in "fields".

    Parameters
    ----------
    data : {pandas.DataFrame, DictOfSeries}
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
    deviants : list
        A list containing the column positions of deviant variables in the input

    """
    var_num = len(data.columns)
    if var_num <= 1:
        return []
    dist_mat = np.zeros((var_num, var_num))
    combs = list(itertools.combinations(range(0, var_num), 2))
    for i, j in combs:
        d_i = data[data.columns[i]]
        d_j = data[data.columns[j]]
        dist = metric(d_i.values, d_j.values)
        dist_mat[i, j] = dist

    condensed = np.abs(dist_mat[tuple(zip(*combs))])
    Z = linkage(condensed, method=linkage_method)
    cluster = fcluster(Z, norm_spread, criterion="distance")
    if population == "variables":
        counts = collections.Counter(cluster)
        pop_num = var_num
    elif population == "samples":
        counts = {cluster[j]: 0 for j in range(0, var_num)}
        for c in range(var_num):
            field = data.columns[c]
            counts[cluster[c]] += data[field].dropna().shape[0]
        pop_num = np.sum(list(counts.values()))
    else:
        raise ValueError(
            "Not a valid normality criteria keyword passed. "
            "Pass either 'variables' or 'population'."
        )
    norm_cluster = -1

    for item in counts.items():
        if item[1] > norm_frac * pop_num:
            norm_cluster = item[0]
            break

    if norm_cluster == -1 or counts[norm_cluster] == pop_num:
        return []
    else:
        return [i for i, x in enumerate(cluster) if x != norm_cluster]


def getFreqDelta(index: pd.Index) -> None | pd.Timedelta:
    """
    Function checks if the passed index is regularly sampled.

    If yes, the according timedelta value is returned,

    If no, ``None`` is returned.

    (``None`` will also be returned for pd.RangeIndex type.)

    """
    delta = getattr(index, "freq", None)
    if delta is None and not index.empty:
        i = pd.date_range(index[0], index[-1], len(index))
        if i.equals(index):
            return i[1] - i[0]
    return delta


def getApply(in_obj, apply_obj, attr_access="__name__", attr_or="apply") -> pd.Series:
    """
    For the repeating task of applying build in (accelerated) methods/funcs (`apply_obj`),
    of rolling/resampling - like objects (`in_obj`) ,
    if those build-ins are available, or pass the method/func to the objects apply-like method, otherwise.

    """
    try:
        out = getattr(in_obj, getattr(apply_obj, attr_access))()
    except AttributeError:
        try:
            # let's try to run it somewhat optimized
            out = getattr(in_obj, attr_or)(apply_obj, raw=True)
        except:
            # did't work out, fallback
            out = getattr(in_obj, attr_or)(apply_obj)

    return out


def statPass(
    datcol: pd.Series,
    stat: Callable[[np.ndarray, pd.Series], float],
    winsz: pd.Timedelta,
    thresh: float,
    comparator: Callable[[CompT, CompT], bool],
    sub_winsz: pd.Timedelta | None = None,
    sub_thresh: float | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Check `datcol`, if it contains chunks of length `window`, exceeding `thresh` with
    regard to `func` and `comparator`:

    (check, if: `comparator`(func`(*chunk*), `thresh`)

    If yes, subsequently check, if all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_window`,
    satisfy, `comparator`(`func`(*sub_chunk*), `sub_thresh`)

    returns boolean series with same index as input series
    """
    stat_parent = datcol.rolling(winsz, min_periods=min_periods)
    stat_parent = getApply(stat_parent, stat)
    exceeds = comparator(stat_parent, thresh)
    if sub_winsz:
        stat_sub = datcol.rolling(sub_winsz)
        stat_sub = getApply(stat_sub, stat)
        min_stat = stat_sub.rolling(winsz - sub_winsz, closed="both").min()
        exceeding_sub = comparator(min_stat, sub_thresh)
        exceeds = exceeding_sub & exceeds

    to_set = pd.Series(False, index=exceeds.index)
    exceeds_df = pd.DataFrame(
        {"g": exceeds.diff().cumsum().values, "ex_val": exceeds.values},
        index=exceeds.index,
    )
    for _, group in exceeds_df.groupby(by="g"):
        if group["ex_val"].iloc[0]:
            # dt-slices include both bounds, so we subtract 1ns
            start = group.index[0] - (winsz - pd.Timedelta("1ns"))
            end = group.index[-1]
            to_set[start:end] = True

    return to_set


def filterKwargs(
    kwargs: dict,
    reserved: Collection,
    inplace: bool = True,
    warn: bool = True,
    msg: str = "",
    stacklevel: int = 3,
) -> dict:
    """
    Filter kwargs (or any dict) by a list of reserved keys.

    Parameters
    ----------
    kwargs : dict
        The dict to filter.

    reserved : list-like
        A list of reserved keywords.

    inplace : bool, default False
        If `False` a copy is returned, otherwise the modified original kwargs.

    warn : bool, default True
        Throw a `RuntimeWarning` with the following text:

    msg : str, default ""
        A text to append to the warnings message.

    stacklevel : int, default 3
        The stacklevel this warning will refer to.
            - `2` : warn at the location this function is called
            - `3` : warn for the function that calls this function (default)

    Returns
    -------
    kwargs: dict
        the modified kwargs or a copy
    """
    if not inplace:
        kwargs = kwargs.copy()
    for key in reserved:
        if warn and key in kwargs:
            warnings.warn(
                f"The keyword {repr(key)} is reserved and will be ignored {msg}",
                RuntimeWarning,
                stacklevel=stacklevel,
            )
        kwargs.pop(key, None)
    return kwargs


from saqc import FILTER_ALL, UNFLAGGED

A = TypeVar("A", np.ndarray, pd.Series)


def isflagged(flagscol: A, thresh: float) -> A:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if not isinstance(thresh, (float, int)):
        raise TypeError(f"thresh must be of type float, not {repr(type(thresh))}")

    if thresh == FILTER_ALL:
        return flagscol > UNFLAGGED

    return flagscol >= thresh


def isunflagged(flagscol: A, thresh: float) -> A:
    return ~isflagged(flagscol, thresh)


def getUnionIndex(obj, default: pd.DatetimeIndex | None = None):
    assert hasattr(obj, "columns")
    if default is None:
        default = pd.DatetimeIndex([])
    indexes = [obj[k].index for k in obj.columns]
    if indexes:
        return functools.reduce(pd.Index.union, indexes).sort_values()
    return default


def getSharedIndex(obj, default: pd.DatetimeIndex | None = None):
    assert hasattr(obj, "columns")
    if default is None:
        default = pd.DatetimeIndex([])
    indexes = [obj[k].index for k in obj.columns]
    if indexes:
        return functools.reduce(pd.Index.intersection, indexes).sort_values()
    return default


def isAllBoolean(obj: Any):
    if not hasattr(obj, "columns"):
        return pd.api.types.is_bool_dtype(obj)
    for c in obj.columns:
        if not pd.api.types.is_bool_dtype(obj[c]):
            return False
    return True
