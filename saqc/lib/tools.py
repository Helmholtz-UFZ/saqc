#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Union, Any, Iterator

import numpy as np
import pandas as pd
import numba as nb
import saqc.lib.ts_operators as ts_ops
import scipy
from functools import reduce, partial
from saqc.lib.types import T, PandasLike

SAQC_OPERATORS = {
    "exp": np.exp,
    "log": np.log,
    "sum": np.sum,
    "var": np.var,
    "std": np.std,
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "first": pd.Series(np.nan, index=pd.DatetimeIndex([])).resample("0min").first,
    "last": pd.Series(np.nan, index=pd.DatetimeIndex([])).resample("0min").last,
    "delta_t": ts_ops.deltaT,
    "id": ts_ops.identity,
    "diff": ts_ops.difference,
    "relDiff": ts_ops.relativeDifference,
    "deriv": ts_ops.derivative,
    "roc": ts_ops.rateOfChange,
    "scale": ts_ops.scale,
    "normScale": ts_ops.normScale,
}


OP_MODULES = {"pd": pd, "np": np, "scipy": scipy}


def evalFuncString(func_string):
    if not isinstance(func_string, str):
        return func_string
    module_dot = func_string.find(".")
    first, *rest = func_string.split(".")
    if rest:
        module = func_string[:module_dot]
        try:
            return reduce(lambda m, f: getattr(m, f), rest, OP_MODULES[first])
        except KeyError:
            availability_list = [f"'{k}' (= {s.__name__})" for k, s in OP_MODULES.items()]
            availability_list = " \n".join(availability_list)
            raise ValueError(
                f'The external-module alias "{module}" is not known to the internal operators dispatcher. '
                f"\n Please select from: \n{availability_list}"
            )

    else:
        if func_string in SAQC_OPERATORS:
            return SAQC_OPERATORS[func_string]
        else:
            availability_list = [f"'{k}' (= {s.__name__})" for k, s in SAQC_OPERATORS.items()]
            availability_list = " \n".join(availability_list)
            raise ValueError(
                f'The external-module alias "{func_string}" is not known to the internal operators '
                f"dispatcher. \n Please select from: \n{availability_list}"
            )


def composeFunction(functions):
    if callable(functions):
        return functions
    functions = toSequence(functions)
    functions = [evalFuncString(f) for f in functions]

    def composed(ts, funcs=functions):
        return reduce(lambda x, f: f(x), reversed(funcs), ts)

    return partial(composed, funcs=functions)


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

    # lets work on numpy data structures for performance reasons
    if isinstance(dates, (pd.DataFrame, pd.Series)):
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


def inferFrequency(data: PandasLike) -> pd.DateOffset:
    return pd.tseries.frequencies.to_offset(pd.infer_freq(data.index))


def combineDataFrames(left: pd.DataFrame, right: pd.DataFrame, fill_value: float = np.nan) -> pd.DataFrame:
    """
    Combine the given DataFrames 'left' and 'right' such that, the
    output is union of the indices and the columns of both. In case
    of duplicated values, 'left' is overwritten by 'right'
    """
    combined = left.reindex(
        index=left.index.union(right.index),
        columns=left.columns.union(right.columns, sort=False),
        fill_value=fill_value,
    )

    for key, values in right.iteritems():
        combined.loc[right.index, key] = values

    return combined


def retrieveTrustworthyOriginal(data: pd.DataFrame, field: str, flagger=None, level: Any = None) -> pd.DataFrame:
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


def assertDataFrame(df: Any, argname: str = "arg", allow_multiindex: bool = True) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{argname} must be of type pd.DataFrame, {type(df)} was given")
    if not allow_multiindex:
        assertSingleColumns(df, argname)
    if not df.columns.is_unique:
        raise TypeError(f"{argname} must have unique columns")


def assertSeries(srs: Any, argname: str = "arg") -> None:
    if not isinstance(srs, pd.Series):
        raise TypeError(f"{argname} must be of type pd.Series, {type(srs)} was given")


def assertPandas(pdlike: PandasLike, argname: str = "arg", allow_multiindex: bool = True) -> None:
    if not isinstance(pdlike, pd.Series) and not isinstance(pdlike, pd.DataFrame):
        raise TypeError(f"{argname} must be of type pd.DataFrame or pd.Series, {type(pdlike)} was given")
    if not allow_multiindex:
        assertSingleColumns(pdlike, argname)


def assertMultiColumns(dfmi: pd.DataFrame, argname: str = "") -> None:
    assertDataFrame(dfmi, argname, allow_multiindex=True)
    if not isinstance(dfmi.columns, pd.MultiIndex):
        raise TypeError(
            f"given pd.DataFrame ({argname}) need to have a muliindex on columns, "
            f"instead it has a {type(dfmi.columns)}"
        )


def assertSingleColumns(df: PandasLike, argname: str = "") -> None:
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        raise TypeError(f"given pd.DataFrame {argname} is not allowed to have a muliindex on columns")


def getFuncFromInput(func):
    """
    Aggregation functions passed by the user, are selected by looking them up in the STRING_2_DICT dictionary -
    But since there are wrappers, that dynamically generate aggregation functions and pass those on ,the parameter
    interfaces must as well be capable of processing real functions passed. This function does that.

    :param func: A key to the STRING_2_FUNC dict, or an actual function
    """
    # if input is a callable - than just pass it:
    if hasattr(func, "__call__"):
        if (func.__name__ == "aggregator") & (func.__module__ == "saqc.funcs.harm_functions"):
            return func
        else:
            raise ValueError("The function you passed is suspicious!")
    else:
        return evalFuncString(func)


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
    target = values[0]

    start = 0
    while True:
        stop = otherIndex(values, start)
        if stop == -1:
            break
        yield pd.Series(data=values[start:stop], index=index[start:stop])
        start = stop
