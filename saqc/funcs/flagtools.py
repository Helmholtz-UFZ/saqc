#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Tuple, Union, Sequence
from typing_extensions import Literal
import pandas as pd
from dateutil.parser import ParserError
import numpy as np
from dios import DictOfSeries

from saqc.constants import BAD, UNFLAGGED
from saqc.core import register, Flags
import warnings

from saqc.core.register import flagging
from saqc.funcs.generic import flagGeneric


@register(mask=[], demask=[], squeeze=["field"])
def forceFlags(
    data: DictOfSeries, field: str, flags: Flags, flag: float = BAD, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Set whole column to a flag value.

    Parameters
    ----------
    data : DictOfSeries
        data container
    field : str
        columns name that holds the data
    flags : saqc.Flags
        flags object
    flag : float, default BAD
        flag to set
    kwargs : dict
        unused

    Returns
    -------
    data : DictOfSeries
    flags : saqc.Flags

    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    flagUnflagged : set flag value at all unflagged positions
    """
    flags[:, field] = flag
    return data, flags


@register(mask=[], demask=[], squeeze=["field"])
def clearFlags(
    data: DictOfSeries, field: str, flags: Flags, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Set whole column to UNFLAGGED.

    Parameters
    ----------
    data : DictOfSeries
        data container
    field : str
        columns name that holds the data
    flags : saqc.Flags
        flags object
    kwargs : dict
        unused

    Returns
    -------
    data : DictOfSeries
    flags : saqc.Flags

    Notes
    -----
    This function ignores the ``dfilter`` keyword, because the data is not relevant
    for processing.
    A warning is triggered if the ``flag`` keyword is given, because the flags are
    always set to `UNFLAGGED`.


    See Also
    --------
    forceFlags : set whole column to a flag value
    flagUnflagged : set flag value at all unflagged positions
    """
    # NOTE: do we really need this?
    if "flag" in kwargs:
        kwargs = {**kwargs}  # copy
        flag = kwargs.pop("flag")
        warnings.warn(f"`flag={flag}` is ignored here.")

    return forceFlags(data, field, flags, flag=UNFLAGGED, **kwargs)


@register(mask=[], demask=[], squeeze=["field"])
def flagUnflagged(
    data: DictOfSeries, field: str, flags: Flags, flag: float = BAD, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function sets a flag at all unflagged positions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    flag : float, default BAD
        flag value to set
    kwargs : Dict
        unused

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    This function ignores the ``dfilter`` keyword, because the data is not relevant
    for processing.

    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    forceFlags : set whole column to a flag value
    """
    unflagged = flags[field].isna() | (flags[field] == UNFLAGGED)
    flags[unflagged, field] = flag
    return data, flags


@register(mask=["field"], demask=["field"], squeeze=["field"])
def flagManual(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    mdata: Union[pd.Series, pd.DataFrame, DictOfSeries, list, np.array],
    method: Literal[
        "left-open", "right-open", "closed", "plain", "ontime"
    ] = "left-open",
    mformat: Literal["start-end", "mflag"] = "start-end",
    mflag: Any = 1,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    mdata : pd.Series, pd.DataFrame, DictOfSeries, str, list or np.ndarray
        The Data determining, wich intervals are to be flagged, or a string, denoting under which field the data is
        accessable.
    method : {'plain', 'ontime', 'left-open', 'right-open', 'closed'}, default 'plain'
        Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
        index.

        * 'plain': mdata must have the same length as data and is projected one-to-one on data.
        * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
        * 'right-open': mdata defines intervals, values are to be projected on.
          The intervals are defined,

          (1) Either, by any two consecutive timestamps t_1 and 1_2 where t_1 is valued with mflag, or by a series,
          (2) Or, a Series, where the index contains in the t1 timestamps nd the values the respective t2 stamps.

          The value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.

        * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.
        * 'closed': like 'right-open', but the projected interval now covers all t with t_1 <= t <= t_2.

    mformat : {"start-end", "mflag"}, default "start-end"

        * "start-end": mdata is a Series, where every entry indicates an interval to-flag. The index defines the left
          bound, the value defines the right bound.
        * "mflag": mdata is an array like, with entries containing 'mflag',where flags shall be set. See documentation
          for examples.

    mflag : scalar
        The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : original data
    flags : modified flags

    Examples
    --------
    An example for mdata

    .. doctest:: ExampleFlagManual

       >>> mdata = pd.Series([1,0,1], index=pd.to_datetime(['2000-02', '2000-03', '2001-05']))
       >>> mdata
       2000-02-01    1
       2000-03-01    0
       2001-05-01    1
       dtype: int64

    On *dayly* data, with the 'ontime' method, only the provided timestamps are used.
    Bear in mind that only exact timestamps apply, any offset will result in ignoring
    the timestamp.

    .. doctest:: ExampleFlagManual

       >>> data = a=pd.Series(0, index=pd.date_range('2000-01-31', '2000-03-02', freq='1D'), name='dailyData')
       >>> qc = saqc.SaQC(data)
       >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='ontime')
       >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
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

    .. doctest:: ExampleFlagManual

       >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='right-open')
       >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
       2000-01-31    False
       2000-02-01    True
       2000-02-02    True
       ..            ..
       2000-02-29    True
       2000-03-01    False
       2000-03-02    False
       Freq: D, dtype: bool

    With the 'left-open' method, backward filling is used:

    .. doctest:: ExampleFlagManual

       >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='left-open')
       >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
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
    # internal not-mflag-value -> cant go for np.nan
    not_mflag = -1 if mflag == 0 else 0
    if isinstance(mdata, str):
        mdata = data[mdata]

    if isinstance(mdata, (pd.DataFrame, DictOfSeries)):
        mdata = mdata[field]

    hasindex = isinstance(mdata, (pd.Series, pd.DataFrame, DictOfSeries))
    if not hasindex:
        if method != "plain":
            raise ValueError("mdata has no index")
        else:
            mdata = pd.Series(mdata, index=dat.index)

    # check, if intervals where passed in format (index:start-time, data:end-time)
    if mformat == "start-end":
        if method in ["plain", "ontime"]:
            raise ValueError(
                "'Start-End' formatting not compatible to 'plain' or 'ontime' methods"
            )
        else:
            mdata = pd.Series(
                not_mflag,
                index=mdata.index.join(pd.DatetimeIndex(mdata.values), how="outer"),
            )
            mdata[::2] = mflag

    # get rid of values that are neither mflag nor not_mflag (for bw-compatibillity mainly)
    mdata[mdata != mflag] = not_mflag

    # evaluate methods
    if method == "plain":
        pass
    # reindex will do the job later
    elif method == "ontime":
        pass

    elif method in ["left-open", "right-open", "closed"]:
        mdata = mdata.drop(mdata.index[mdata.diff() == 0])
        app_entry = pd.Series(mdata[-1], dat.index.shift(freq="1min")[-1:])
        mdata = mdata.reindex(dat.index.union(mdata.index))

        if method == "right-open":
            mdata = mdata.ffill()

        if method == "left-open":
            mdata = (
                mdata.replace({mflag: not_mflag, not_mflag: mflag})
                .append(app_entry)
                .bfill()
            )

        if method == "closed":
            mdata[mdata.ffill() == mflag] = mflag
            mdata.replace({not_mflag: mflag}, inplace=True)
    else:
        raise ValueError(method)

    mask = mdata == mflag
    mask = mask.reindex(dat.index).fillna(False)

    flags[mask, field] = flag
    return data, flags


@flagging()
def flagDummy(
    data: DictOfSeries, field: str, flags: Flags, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function does nothing but returning data and flags.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
    """
    return data, flags


@register(
    mask=[],
    demask=[],
    squeeze=["target"],
    handles_target=True,
)
def transferFlags(
    data: DictOfSeries,
    field: str | Sequence[str],
    flags: Flags,
    target: str | Sequence[str],
    **kwargs,
):
    """
    Transfer Flags of one variable to another.

    Parameters
    ----------

    data : {pd.DataFrame, dios.DictOfSeries}
        data

    field : str or List of str
       Variable or list of variables, the flags of which are to be transferred.

    flags : {pd.DataFrame, dios.DictOfSeries, saqc.flagger}
        Flags or flagger object

    target : str or List of str
        Variable or list of variables, the flags of `field` are to be transferred to.

    See Also
    --------
    * :py:meth:`saqc.SaQC.flagGeneric`

    Examples
    --------
    First, generate some data with some flags:

    .. doctest:: exampleTransfer

       >>> import pandas as pd #doctest:+SKIP
       >>> import saqc #doctest:+SKIP
       >>> data = pd.DataFrame({'a':[1,2], 'b':[1,2], 'c':[1,2]})
       >>> qc = saqc.SaQC(data)
       >>> qc = qc.flagRange('a', max=1.5)
       >>> qc.flags
       columns      a    b    c
       0         -inf -inf -inf
       1        255.0 -inf -inf

    Now we can project the flag from `a` to `b` via

    .. doctest:: exampleTransfer

       >>> qc = qc.transferFlags('a', target='b')
       >>> qc.flags
       columns      a      b    c
       0         -inf   -inf -inf
       1        255.0  255.0 -inf

    You can skip the explicit target parameter designation:

    .. doctest:: exampleTransfer

       >>> qc = qc.transferFlags('a', 'b')

    To project the flags of `a` to both the variables `b` and `c` in one call, align the field and target variables in
    2 lists:

    .. doctest:: exampleTransfer

       >>> qc = qc.transferFlags(['a','a'], ['b', 'c'])
       >>> qc.flags
       columns      a      b      c
       0         -inf   -inf   -inf
       1        255.0  255.0  255.0
    """

    data, flags = flagGeneric(
        data, field, flags, target=target, func=lambda x: isflagged(x), **kwargs
    )
    return data, flags
