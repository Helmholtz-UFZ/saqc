#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Tuple, Optional, Union
from typing_extensions import Literal

import pandas as pd

from dios import DictOfSeries

from saqc.common import *
from saqc.lib.types import *
from saqc.core.register import register
from saqc.flagger import Flagger


@register(masking='field', module="flagtools")
def clearFlags(data: DictOfSeries, field: ColumnName, flagger: Flagger, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    flagger = flagger.clearFlags(field, **kwargs)
    return data, flagger


@register(masking='field', module="flagtools")
def forceFlags(data: DictOfSeries, field: ColumnName, flagger: Flagger, flag: Any, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    flagger = flagger.clearFlags(field).setFlags(field, flag=flag, inplace=True, **kwargs)
    return data, flagger


@register(masking='field', module="flagtools")
def flagDummy(data: DictOfSeries, field: ColumnName, flagger: Flagger,  **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    Function does nothing but returning data and flagger.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
    """
    return data, flagger


@register(masking='field', module="flagtools")
def flagForceFail(data: DictOfSeries, field: ColumnName, flagger: Flagger, **kwargs):
    """
    Function raises a runtime error.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.

    """
    raise RuntimeError("Works as expected :D")


@register(masking='field', module="flagtools")
def flagUnflagged(data: DictOfSeries, field: ColumnName, flagger: Flagger, flag: Optional[Any]=None, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    Function sets the GOOD flag to all values flagged better then GOOD.
    If there is an entry 'flag' in the kwargs dictionary passed, the
    function sets the kwargs['flag'] flag to all values flagged better kwargs['flag']

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.
    kwargs : Dict
        If kwargs contains 'flag' entry, kwargs['flag] is set, if no entry 'flag' is present,
        'UNFLAGGED' is set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
    """

    flag = GOOD if flag is None else flag
    flagger = flagger.setFlags(field, flag=flag, **kwargs)
    return data, flagger


@register(masking='field', module="flagtools")
def flagGood(data: DictOfSeries, field: ColumnName, flagger: Flagger, flag: Optional[Any]=None, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    Function sets the GOOD flag to all values flagged better then GOOD.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.

    """
    return flagUnflagged(data, field, flagger, flag=flag, **kwargs)


@register(masking='field', module="flagtools")
def flagManual(
        data: DictOfSeries, field: ColumnName, flagger: Flagger,
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries],
        mflag: Any = 1,
        method=Literal["plain", "ontime", "left-open", "right-open"],
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
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
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.
    mdata : {pd.Series, pd.Dataframe, DictOfSeries}
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
