#! /usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from inspect import signature

import dios
import numpy as np
import pandas as pd
import scipy
import itertools
import collections
import numba
from mlxtend.evaluate import permutation_test
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster


from saqc.lib.tools import groupConsecutives, detectDeviants
from saqc.lib.tools import groupConsecutives, seasonalMask
from saqc.funcs.proc_functions import proc_fork, proc_drop, proc_projectFlags
from saqc.funcs.modelling import modelling_mask

from saqc.core.register import register
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

    sig = signature(func)
    args = []
    for k, v in sig.parameters.items():
        k = field if k == "this" else k
        if k not in data:
            raise NameError(f"variable '{k}' not found")
        args.append(data[k])

    globs = {
        "isflagged": partial(_dslIsFlagged, flagger),
        "ismissing": lambda var: ((var == nodata) | pd.isnull(var)),
        "mask": lambda cond: data[cond.name].mask(cond),
        "this": field,
        "NODATA": nodata,
        "GOOD": flagger.GOOD,
        "BAD": flagger.BAD,
        "UNFLAGGED": flagger.UNFLAGGED,
        **ENVIRONMENT,
    }
    func.__globals__.update(globs)
    return func(*args)


@register(masking='all')
def procGeneric(data, field, flagger, func, nodata=np.nan, **kwargs):
    """
    generate/process data with generically defined functions.

    The functions can depend on on any of the fields present in data.

    Formally, what the function does, is the following:

    1.  Let F be a Callable, depending on fields f_1, f_2,...f_K, (F = F(f_1, f_2,...f_K))
        Than, for every timestamp t_i that occurs in at least one of the timeseries data[f_j] (outer join),
        The value v_i is computed via:
        v_i = data([f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]), if all data[f_j][t_i] do exist
        v_i = `nodata`, if at least one of the data[f_j][t_i] is missing.
    2.  The result is stored to data[field] (gets generated if not present)

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, where you want the result from the generic expressions processing to be written to.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    func : Callable
        The data processing function with parameter names that will be
        interpreted as data column entries.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        The shape of the data may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        The flags shape may have changed relatively to the input flagger.

    Examples
    --------
    Some examples on what to pass to the func parameter:
    To compute the sum of the variables "temperature" and "uncertainty", you would pass the function:

    >>> lambda temperature, uncertainty: temperature + uncertainty

    You also can pass numpy and pandas functions:

    >>> lambda temperature, uncertainty: np.round(temperature) * np.sqrt(uncertainty)

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


@register(masking='all')
def flagGeneric(data, field, flagger, func, nodata=np.nan, **kwargs):
    """
    a function to flag a data column by evaluation of a generic expression.

    The expression can depend on any of the fields present in data.

    Formally, what the function does, is the following:

    Let X be an expression, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
    Than for every timestamp t_i in data[field]:
    data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.

    Note, that all value series included in the expression to evaluate must be labeled identically to field.

    Note, that the expression is passed in the form of a Callable and that this callables variable names are
    interpreted as actual names in the data header. See the examples section to get an idea.

    Note, that all the numpy functions are available within the generic expressions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, where you want the result from the generic expressions evaluation to be projected
        to.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    func : Callable
        The expression that is to be evaluated is passed in form of a callable, with parameter names that will be
        interpreted as data column entries. The Callable must return an boolen array like.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    Examples
    --------
    Some examples on what to pass to the func parameter:
    To flag the variable `field`, if the sum of the variables
    "temperature" and "uncertainty" is below zero, you would pass the function:

    >>> lambda temperature, uncertainty: temperature + uncertainty < 0

    There is the reserved name 'This', that always refers to `field`. So, to flag field if field is negative, you can
    also pass:

    >>> lambda this: this < 0

    If you want to make dependent the flagging from flags already present in the data, you can use the built-in
    ``isflagged`` method. For example, to flag the 'temperature', if 'level' is flagged, you would use:

    >>> lambda level: isflagged(level)

    You can furthermore specify a flagging level, you want to compare the flags to. For example, for flagging
    'temperature', if 'level' is flagged at a level named 'doubtfull' or worse, use:

    >>> lambda level: isflagged(level, flag='doubtfull', comparator='<=')

    If you are unsure about the used flaggers flagging level names, you can use the reserved key words BAD, UNFLAGGED
    and GOOD, to refer to the worst (BAD), best(GOOD) or unflagged (UNFLAGGED) flagging levels. For example.

    >>> lambda level: isflagged(level, flag=UNFLAGGED, comparator='==')

    Your expression also is allowed to include pandas and numpy functions

    >>> lambda level: np.sqrt(level) > 7
    """
    # NOTE:
    # The naming of the func parameter is pretty confusing
    # as it actually holds the result of a generic expression
    mask = _execGeneric(flagger, data, func, field, nodata).squeeze()
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")

    # if flagger.getFlags(field).empty:
    #     flagger = flagger.merge(
    #         flagger.initFlags(
    #             data=pd.Series(name=field, index=mask.index, dtype=np.float64)))
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register(masking='field')
def flagRange(data, field, flagger, min=-np.inf, max=np.inf, **kwargs):
    """
    Function flags values not covered by the closed interval [`min`, `max`].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    min : float
        Lower bound for valid data.
    max : float
        Upper bound for valid data.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    # using .values is very much faster
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger



@register(masking='field')
def flagMissing(data, field, flagger, nodata=np.nan, **kwargs):
    """
    The function flags all values indicating missing data.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    nodata : any, default np.nan
        A value that defines missing data.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    datacol = data[field]
    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol == nodata

    flagger = flagger.setFlags(field, loc=mask, **kwargs)
    return data, flagger


@register(masking='field')
def flagSesonalRange(
        data, field, flagger, min, max, startmonth=1, endmonth=12, startday=1, endday=31, **kwargs,
):
    """
    Function applies a range check onto data chunks (seasons).

    The data chunks to be tested are defined by annual seasons that range from a starting date,
    to an ending date, wheras the dates are defined by month and day number.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    min : float
        Lower bound for valid data.
    max : float
        Upper bound for valid data.
    startmonth : int
        Starting month of the season to flag.
    endmonth : int
        Ending month of the season to flag.
    startday : int
        Starting day of the season to flag.
    endday : int
        Ending day of the season to flag

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """
    if data[field].empty:
        return data, flagger

    newfield = f"{field}_masked"
    start = f"{startmonth:02}-{startday:02}T00:00:00"
    end = f"{endmonth:02}-{endday:02}T00:00:00"

    data, flagger = proc_fork(data, field, flagger, suffix="_masked")
    data, flagger = modelling_mask(data, newfield, flagger, mode='seasonal', season_start=start, season_end=end,
                                   include_bounds=True)
    data, flagger = flagRange(data, newfield, flagger, min=min, max=max, **kwargs)
    data, flagger = proc_projectFlags(data, field, flagger, method='match', source=newfield)
    data, flagger = proc_drop(data, newfield, flagger)
    return data, flagger


@register(masking='field')
def clearFlags(data, field, flagger, **kwargs):
    flagger = flagger.clearFlags(field, **kwargs)
    return data, flagger


@register(masking='field')
def forceFlags(data, field, flagger, flag, **kwargs):
    flagger = flagger.clearFlags(field).setFlags(field, flag=flag, inplace=True, **kwargs)
    return data, flagger


@register(masking='field')
def flagIsolated(
        data, field, flagger, gap_window, group_window, **kwargs,
):
    """
    The function flags arbitrary large groups of values, if they are surrounded by sufficiently
    large data gaps. A gap is defined as group of missing and/or flagged values.

    A series of values x_k,x_(k+1),...,x_(k+n), with associated timestamps t_k,t_(k+1),...,t_(k+n),
    is considered to be isolated, if:

    1. t_(k+1) - t_n < `group_window`
    2. None of the x_j with 0 < t_k - t_j < `gap_window`, is valid or unflagged (preceeding gap).
    3. None of the x_j with 0 < t_j - t_(k+n) < `gap_window`, is valid or unflagged (succeding gap).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    gap_window :
        The minimum size of the gap before and after a group of valid values, making this group considered an
        isolated group. See condition (2) and (3)
    group_window :
        The maximum temporal extension allowed for a group that is isolated by gaps of size 'gap_window',
        to be actually flagged as isolated group. See condition (1).

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

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
                left = mask[start - gap_window: start].iloc[:-1]
                if left.all():
                    right = mask[stop: stop + gap_window].iloc[1:]
                    if right.all():
                        flags[start:stop] = True

    flagger = flagger.setFlags(field, flags, **kwargs)

    return data, flagger


@register(masking='field')
def flagDummy(data, field, flagger, **kwargs):
    """
    Function does nothing but returning data and flagger.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
    """
    return data, flagger


@register(masking='field')
def flagForceFail(data, field, flagger, **kwargs):
    """
    Function raises a runtime error.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.

    """
    raise RuntimeError("Works as expected :D")


@register(masking='field')
def flagUnflagged(data, field, flagger, **kwargs):
    """
    Function sets the flagger.GOOD flag to all values flagged better then flagger.GOOD.
    If there is an entry 'flag' in the kwargs dictionary passed, the
    function sets the kwargs['flag'] flag to all values flagged better kwargs['flag']

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    kwargs : Dict
        If kwargs contains 'flag' entry, kwargs['flag] is set, if no entry 'flag' is present,
        'flagger.UNFLAGGED' is set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
    """

    flag = kwargs.pop('flag', flagger.GOOD)
    flagger = flagger.setFlags(field, flag=flag, **kwargs)
    return data, flagger


@register(masking='field')
def flagGood(data, field, flagger, **kwargs):
    """
    Function sets the flagger.GOOD flag to all values flagged better then flagger.GOOD.

    Parameters
    ----------
    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.

    """
    kwargs.pop('flag', None)
    return flagUnflagged(data, field, flagger, **kwargs)


@register(masking='field')
def flagManual(data, field, flagger, mdata, mflag: Any = 1, method="plain", **kwargs):
    """
    Flag data by given, "manually generated" data.

    The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
    The format of mdata can be an indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
    but also can be a plain list- or array-like.
    How indexed mdata is aligned to data is specified via the `method` parameter.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    mdata : {pd.Series, pd.Dataframe, DictOfSeries, str}
        The "manually generated" data
    mflag : scalar
        The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
    method : {'plain', 'ontime', 'left-open', 'right-open'}, default plain
        Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
        index.

        * 'plain': mdata must have the same length as data and is projected one-to-one on data.
        * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
        * 'right-open': mdata defines intervals, values are to be projected on.
          The intervals are defined by any two consecutive timestamps t_1 and 1_2 in mdata.
          the value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.
        * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.

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


@register(masking='all')
def flagCrossScoring(data, field, flagger, fields, thresh, cross_stat='modZscore', **kwargs):
    """
    Function checks for outliers relatively to the "horizontal" input data axis.

    For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
    for outlier detection:

    1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
       :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
    2. for every :math:`0 <= i <= K`, the value
       :math:`m_j = median(\\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\\})` is calculated
    2. for every :math:`0 <= i <= K`, the set
       :math:`\\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\\}` is tested for outliers with the
       specified method (`cross_stat` parameter).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    thresh : float
        Threshold which the outlier score of an value must exceed, for being flagged an outlier.
    cross_stat : {'modZscore', 'Zscore'}, default 'modZscore'
        Method used for calculating the outlier scores.

        * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
        * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
          See References [1]

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """

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

@register(masking='all')
def flagDriftFromNorm(data, field, flagger, fields, segment_freq, norm_spread, norm_frac=0.5,
                      metric=lambda x, y: scipy.spatial.distance.pdist(np.array([x, y]),
                                                                       metric='cityblock') / len(x),
                      linkage_method='single', **kwargs):
    """
    The function flags value courses that significantly deviate from a group of normal value courses.

    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".

    See the Notes section for a more detailed presentation of the algorithm

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining which variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    Notes
    -----
    following steps are performed for every data "segment" of length `segment_freq` in order to find the
    "abnormal" data:

    1. Calculate the distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `fields`. (with :math:`d`
       denoting the distance function
       passed to the parameter `metric`.
    2. Calculate a dendogram with a hierarchical linkage algorithm, specified by the parameter `linkage_method`.
    3. Flatten the dendogram at the level, the agglomeration costs exceed the value given by the parameter `norm_spread`
    4. check if there is a cluster containing more than `norm_frac` percentage of the variables in fields.

        1. if yes: flag all the variables that are not in that cluster (inside the segment)
        2. if no: flag nothing

    The main parameter giving control over the algorithms behavior is the `norm_spread` parameter, that determines
    the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in every
    linkage step.
    For singleton clusters, that costs just equal half the distance, the timeseries in the clusters, have to
    each other. So, no timeseries can be clustered together, that are more then
    2*`norm_spread` distanted from each other.
    When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is
    calculated according to the linkage method specified by `linkage_method`. By default, it is the minimum distance,
    the members of the clusters have to each other.
    Having that in mind, it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).

    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """

    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)
    segments = data_to_flag.groupby(pd.Grouper(freq=segment_freq))
    for segment in segments:
        if segment[1].shape[0] <= 1:
            continue
        drifters = detectDeviants(segment[1], metric, norm_spread, norm_frac, linkage_method, 'variables')

        for var in drifters:
            flagger = flagger.setFlags(fields[var], loc=segment[1].index, **kwargs)

    return data, flagger

@register(masking='all')
def flagDriftFromReference(data, field, flagger, fields, segment_freq, thresh,
                      metric=lambda x, y: scipy.spatial.distance.pdist(np.array([x, y]),
                                                                    metric='cityblock')/len(x),
                       **kwargs):
    """
    The function flags value courses that deviate from a reference course by a margin exceeding a certain threshold.

    The deviation is measured by the distance function passed to parameter metric.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    thresh : float
        The threshod by wich normal variables can deviate from the reference variable at max.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    Notes
    -----
    it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).
    """

    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)
    if field not in fields:
        fields.append(field)
    var_num = len(fields)
    segments = data_to_flag.groupby(pd.Grouper(freq=segment_freq))

    for segment in segments:

        if segment[1].shape[0] <= 1:
            continue
        for i in range(var_num):
            dist = metric(segment[1].iloc[:, i].values, segment[1].loc[:, field].values)
            if dist > thresh:
                flagger = flagger.setFlags(fields[i], loc=segment[1].index, **kwargs)

    return data, flagger


@register(masking='all')
def flagDriftScale(data, field, flagger, fields_scale1, fields_scale2, segment_freq, norm_spread, norm_frac=0.5,
                      metric=lambda x, y: scipy.spatial.distance.pdist(np.array([x, y]),
                                                                                    metric='cityblock')/len(x),
                      linkage_method='single', **kwargs):


    """
    The function linearly rescales one set of variables to another set of variables with a different scale and then
    flags value courses that significantly deviate from a group of normal value courses.

    The two sets of variables can be linearly scaled one to another and hence the scaling transformation is performed
    via linear regression: A linear regression is performed on each pair of variables giving a slope and an intercept.
    The transformation is then calculated a the median of all the calculated slopes and intercepts.

    Once the transformation is performed, the function flags those values, that deviate from a group of normal values.
    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.
    fields_scale1 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 1.
    fields_scale2 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 2.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """

    fields = fields_scale1 + fields_scale2
    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)

    convert_slope = []
    convert_intercept = []

    for field1 in fields_scale1:
        for field2 in fields_scale2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_to_flag[field1], data_to_flag[field2])
            convert_slope.append(slope)
            convert_intercept.append(intercept)

    factor_slope = np.median(convert_slope)
    factor_intercept = np.median(convert_intercept)

    dat = dios.DictOfSeries()
    for field1 in fields_scale1:
        dat[field1] = factor_intercept + factor_slope * data_to_flag[field1]
    for field2 in fields_scale2:
        dat[field2] = data_to_flag[field2]

    dat_to_flag = dat[fields].to_df()

    segments = dat_to_flag.groupby(pd.Grouper(freq=segment_freq))
    for segment in segments:
        if segment[1].shape[0] <= 1:
            continue
        drifters = detectDeviants(segment[1], metric, norm_spread, norm_frac, linkage_method, 'variables')
        for var in drifters:
            flagger = flagger.setFlags(fields[var], loc=segment[1].index, **kwargs)

    return data, flagger
