#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import BAD, FILTER_ALL, UNFLAGGED
from saqc.core import DictOfSeries, flagging, register
from saqc.lib.checking import validateChoice, validateWindow
from saqc.lib.tools import isflagged, isunflagged, toSequence

if TYPE_CHECKING:
    from saqc import SaQC


class FlagtoolsMixin:
    @flagging()
    def flagDummy(self: "SaQC", field: str, **kwargs) -> "SaQC":
        """
        Function does nothing but returning data and flags.
        """
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def forceFlags(self: "SaQC", field: str, flag: float = BAD, **kwargs) -> "SaQC":
        """
        Set whole column to a flag value.

        See Also
        --------
        clearFlags : set whole column to UNFLAGGED
        flagUnflagged : set flag value at all unflagged positions
        """
        self._flags[:, field] = flag
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def clearFlags(self: "SaQC", field: str, **kwargs) -> "SaQC":
        """
        Set whole column to UNFLAGGED.

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the data
        is not relevant for processing.
        A warning is triggered if the ``flag`` keyword is given, because
        the flags are always set to `UNFLAGGED`.

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

        return self.forceFlags(field, flag=UNFLAGGED, **kwargs)

    @register(mask=[], demask=[], squeeze=["field"])
    def flagUnflagged(self: "SaQC", field: str, flag: float = BAD, **kwargs) -> "SaQC":
        """
        Function sets a flag at all unflagged positions.

        See Also
        --------
        clearFlags : set whole column to UNFLAGGED
        forceFlags : set whole column to a flag value

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the
        data is not relevant for processing.
        """
        unflagged = self._flags[field].isna() | (self._flags[field] == UNFLAGGED)
        self._flags[unflagged, field] = flag
        return self

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def flagManual(
        self: "SaQC",
        field: str,
        mdata: str | pd.Series | np.ndarray | list | pd.DataFrame | DictOfSeries,
        method: Literal[
            "left-open", "right-open", "closed", "plain", "ontime"
        ] = "left-open",
        mformat: Literal["start-end", "mflag"] = "start-end",
        mflag: Any = 1,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Include flags listed in external data.

        The method allows to integrate pre-existing flagging information.

        Parameters
        ----------
        mdata :
            Determines which values or intervals will be flagged. Supported input types:

            * ``pd.Series``: Needs a datetime index and values of type:

              - datetime, for :py:attr:`method` values ``"right-closed"``, ``"left-closed"``, ``"closed"``
              - or any scalar, for :py:attr:`method` values ``"plain"``, ``"ontime"``

            * ``str``: Variable holding the manual flag information.
            * ``pd.DataFrame``, ``DictOfSeries``: Need to provide a ``pd.Series`` with column name
              :py:attr:`field`.
            * ``list``, ``np.ndarray``: Only supported with :py:attr:`method` value ``"plain"`` and
              :py:attr:`mformat` value ``"mflag"``
        method :
            Defines how :py:attr:`mdata` is projected to data:

            * ``"plain"``: :py:attr:`mdata` must have the same length as :py:attr:`field`, flags
              are set, where the values in :py:attr:`mdata` equal :py:attr:`mflag`.
            * ``"ontime"``: Expects datetime indexed :py:attr:`mdata` (types ``pd.Series``,
              ``pd.DataFrame``, ``DictOfSeries``). Flags are set, where the values in
              :py:attr:`mdata` equal :py:attr:`mflag` and the indices of :py:attr:`field` and
              :py:attr:`mdata` match.
            * ``"right-open"``: Expects datetime indexed :py:attr:`mdata`, which will be interpreted
              as a number of time intervals ``t_1, t_2``. Flags are set to all timestamps ``t`` of
              :py:attr:`field` with ``t_1 <= t < t_2``.
            * ``"left-open"``: like ``"right-open"``, but the interval covers all ``t`` with
              ``t_1 < t <= t_2``.
            * ``"closed"``: like ``"right-open"``, but the interval now covers all ``t`` with
              ``t_1 <= t <= t_2``.
        mformat :
            Controls the interval definition in :py:attr:`mdata` (see examples):

            * ``"start-end"``: expects datetime indexed :py:attr:`mdata` (types ``pd.Series``,
              ``pd.DataFrame``, ``DictOfSeries``) with values of type datetime. Each
              index-value pair is interpreted as an interval to flag, the index defines the
              left bound, the respective value the right bound.
            * ``"mflag"``:

              - :py:attr:`mdata` of type ``pd.Series``, ``pd.DataFrame``, ``DictOfSeries``:
                Two successive index values ``i_1, i_2`` will be interpreted as an interval
                ``t_1, t_2`` to flag, if the value of ``t_1`` equals :py:attr:`mflag`
              - :py:attr:`mdata` of type ``list``, ``np.ndarray``: Flags all :py:attr:`field`
                where :py:attr:`mdata` euqals :py:attr:`mflag`.

        mflag :
            Value in :py:attr:`mdata` indicating that a flag should be set at the respective
            position, timestamp or interval. Ignored if :py:attr:`mformat` is set to ``"start-end"``.


        Examples
        --------
        Usage of :py:attr:`mdata`

        .. doctest:: ExampleFlagManual

           >>> import saqc
           >>> mdata = pd.Series([1, 0, 1], index=pd.to_datetime(['2000-02-01', '2000-03-01', '2000-05-01']))
           >>> mdata
           2000-02-01    1
           2000-03-01    0
           2000-05-01    1
           dtype: int64

        On *daily* data, with :py:attr:`method` ``"ontime"``, only the provided timestamps
        are used. Only exact matches apply, offsets will be ignored.

        .. doctest:: ExampleFlagManual

           >>> data = pd.Series(0, index=pd.to_datetime(['2000-01-31', '2000-02-01', '2000-02-02', '2000-03-01', '2000-05-01']), name='daily_data')
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mflag', method='ontime')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02    False
           2000-03-01    False
           2000-05-01     True
           dtype: bool

        With :py:attr:`method` ``"right-open"`` , :py:attr:`mdata` is forward filled:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mflag', method='right-open')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02     True
           2000-03-01    False
           2000-05-01     True
           dtype: bool

        With :py:attr:`method` ``"left-open"`` , :py:attr:`mdata` is backward filled:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mflag', method='left-open')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02     True
           2000-03-01     True
           2000-05-01     True
           dtype: bool
        """
        validateChoice(
            method, "method", ["left-open", "right-open", "closed", "plain", "ontime"]
        )
        validateChoice(mformat, "mformat", ["start-end", "mflag"])

        dat = self._data[field]
        # internal not-mflag-value -> cant go for np.nan
        not_mflag = -1 if mflag == 0 else 0
        if isinstance(mdata, str):
            mdata = self._data[mdata]

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
                    "'start-end'-format is not compatible "
                    "with methods 'plain' or 'ontime'"
                )
            else:
                mdata = pd.Series(
                    not_mflag,
                    index=mdata.index.join(pd.DatetimeIndex(mdata.values), how="outer"),
                )
                mdata[::2] = mflag

        # get rid of values that are neither mflag
        # nor not_mflag (for bw-compatibility mainly)
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
                mdata = pd.concat(
                    [mdata.replace({mflag: not_mflag, not_mflag: mflag}), app_entry]
                ).bfill()

            if method == "closed":
                mdata[mdata.ffill() == mflag] = mflag
                mdata.replace({not_mflag: mflag}, inplace=True)
        else:
            raise ValueError(method)

        mask = mdata == mflag
        mask = mask.reindex(dat.index).fillna(False)

        self._flags[mask, field] = flag
        return self

    @register(
        mask=[],
        demask=[],
        squeeze=["target"],
        handles_target=True,  # function defines a target parameter, so it needs to handle it
    )
    def transferFlags(
        self: "SaQC",
        field: str,
        target: str,
        **kwargs,
    ) -> "SaQC":
        """
        Transfer Flags of one variable to another.

        .. deprecated:: 2.4.0
           Use :py:meth:`~saqc.SaQC.concatFlags` with ``method="match"`` and ``squeeze=False`` instead.

        See Also
        --------
        * :py:meth:`saqc.SaQC.flagGeneric`
        * :py:meth:`saqc.SaQC.concatFlags`

        Examples
        --------
        First, generate some data with some flags:

        .. doctest:: exampleTransfer

           >>> import saqc
           >>> data = pd.DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [1, 2]})
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagRange('a', max=1.5)
           >>> qc.flags.to_pandas()
                  a    b    c
           0   -inf -inf -inf
           1  255.0 -inf -inf

        Now we can project the flag from `a` to `b` via

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags('a', target='b')
           >>> qc.flags.to_pandas()
                  a      b    c
           0   -inf   -inf -inf
           1  255.0  255.0 -inf

        You can skip the explicit target parameter designation:

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags('a', 'b')

        To project the flags of `a` to both the variables `b` and `c`
        in one call, align the field and target variables in 2 lists:

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags(['a','a'], ['b', 'c'])
           >>> qc.flags.to_pandas()
                  a      b      c
           0   -inf   -inf   -inf
           1  255.0  255.0  255.0
        """
        import warnings

        warnings.warn(
            f"The method 'transferFlags' is deprecated and will be removed "
            f"in version 2.5 of SaQC. Please use `SaQC.concatFlags(field={field}, "
            f"target={target}, method='match', squeeze=False)` instead",
            DeprecationWarning,
        )
        return self.concatFlags(field, target=target, method="match", squeeze=False)

    @flagging()
    def propagateFlags(
        self: "SaQC",
        field: str,
        window: str | int,
        method: Literal["ffill", "bfill"] = "ffill",
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Flag values before or after flags set by the last test.

        Parameters
        ----------
        window :
            Size of the repetition window. An integer defines the exact
            number of repetitions, strings are interpreted as time offsets
            to fill with.

        method :
            Direction of repetetion. With "ffill" the subsequent values
            receive the flag to repeat, with "bfill" the previous values.

        Examples
        --------
        First, generate some data and some flags:

        .. doctest:: propagateFlags

           >>> import saqc
           >>> data = pd.DataFrame({"a": [-3, -2, -1, 0, 1, 2, 3]})
           >>> flags = pd.DataFrame({"a": [-np.inf, -np.inf, -np.inf, 255.0, -np.inf, -np.inf, -np.inf]})
           >>> qc = saqc.SaQC(data=data, flags=flags)
           >>> qc.flags["a"]
           0     -inf
           1     -inf
           2     -inf
           3    255.0
           4     -inf
           5     -inf
           6     -inf
           dtype: float64

        Now, to repeat the flag '255.0' two times in direction of ascending
        indices, execute:

        .. doctest:: propagateFlags

           >>> qc.propagateFlags('a', window=2, method="ffill").flags["a"]
           0     -inf
           1     -inf
           2     -inf
           3    255.0
           4    255.0
           5    255.0
           6     -inf
           dtype: float64

        Choosing "bfill" will result in

        .. doctest:: propagateFlags

           >>> qc.propagateFlags('a', window=2, method="bfill").flags["a"]
           0     -inf
           1    255.0
           2    255.0
           3    255.0
           4     -inf
           5     -inf
           6     -inf
           dtype: float64

        If an explicit flag is passed, it will be used to fill the
        repetition window

        .. doctest:: propagateFlags

           >>> qc.propagateFlags('a', window=2, method="bfill", flag=111).flags["a"]
           0     -inf
           1    111.0
           2    111.0
           3    255.0
           4     -inf
           5     -inf
           6     -inf
           dtype: float64
        """
        validateWindow(window)
        validateChoice(method, "method", ["bfill", "ffill"])

        # get the last history column
        hc = self._flags.history[field].hist.iloc[:, -1].astype(float)

        if method == "bfill":
            hc = hc[::-1]

        # get dfilter from meta or get of rid of this and
        # consider everything != np.nan as flag
        flagged = isflagged(hc, dfilter)

        mask = (
            flagged.rolling(window, min_periods=1, closed="left")
            .max()
            .fillna(0)
            .astype(bool)
        )

        if method == "bfill":
            mask = mask[::-1]
        mask = isunflagged(self._flags[field], thresh=dfilter) & mask

        self._flags[mask, field] = flag

        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=True,
    )
    def andGroup(
        self: "SaQC",
        field: str | list[str],
        group: Sequence["SaQC"] | dict["SaQC", str | Sequence[str]] | None = None,
        target: str | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag all values, if all the given ``field`` values are already flagged.

        Parameters
        ----------
        group:
            A collection of ``SaQC`` objects to check for flags, defaults to the current object.

            1. If given as a sequence of ``SaQC`` objects, all objects are checked for flags of a
               variable named :py:attr:`field`.
            2. If given as dictionary the keys are interpreted as ``SaQC`` objects and the corresponding
               values as variables of the respective ``SaQC`` object to check for flags.
        """
        return _groupOperation(
            base=self,
            field=field,
            target=target,
            func=operator.and_,
            group=group,
            flag=flag,
            **kwargs,
        )

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=True,
    )
    def orGroup(
        self: "SaQC",
        field: str | list[str],
        group: Sequence["SaQC"] | dict["SaQC", str | Sequence[str]] | None = None,
        target: str | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag all values, if at least one of the given ``field`` values is already flagged.

        Parameters
        ----------
        group:
            A collection of ``SaQC`` objects to check for flags, defaults to the current object.

            1. If given as a sequence of ``SaQC`` objects, all objects are checked for flags of a
               variable named :py:attr:`field`.
            2. If given as dictionary the keys are interpreted as ``SaQC`` objects and the corresponding
               values as variables of the respective ``SaQC`` object to check for flags.
        """
        return _groupOperation(
            base=self,
            field=field,
            target=target,
            func=operator.or_,
            group=group,
            flag=flag,
            **kwargs,
        )


def _groupOperation(
    base: "SaQC",
    field: str | list[str],
    func: Callable[[pd.Series, pd.Series], pd.Series],
    group: Sequence["SaQC"] | dict["SaQC", str | Sequence[str]] | None = None,
    target: str | list[str] | None = None,
    flag: float = BAD,
    **kwargs,
) -> "SaQC":
    if target is None:
        target = field
    field, target = toSequence(field), toSequence(target)

    if len(target) != 1 and len(target) != len(field):
        raise ValueError(
            "'target' needs to be a string or a sequence of the same length as 'field'"
        )

    # harmonise `group` to type dict[SaQC, list[str]]
    if group is None:
        group = {base: field}
    if not isinstance(group, dict):
        group = {base if isinstance(qc, str) else qc: field for qc in group}
    for k, v in group.items():
        group[k] = toSequence(v)

    # generate mask
    mask = pd.Series(dtype=bool)
    dfilter = kwargs.get("dfilter", FILTER_ALL)
    for qc, fields in group.items():
        if set(field) - qc._flags.keys():
            raise KeyError(
                f"one or more variable(s) in {field} are missing in given SaQC object"
            )
        for f in fields:
            flagged = isflagged(qc._flags[f], thresh=dfilter)
            if mask.empty:
                mask = flagged
            mask = func(mask, flagged)

    # initialize target(s)
    if len(target) == 1:
        if target[0] not in base._data:
            base._data[target[0]] = pd.Series(np.nan, index=mask.index, name=target[0])
            base._flags[target[0]] = pd.Series(np.nan, index=mask.index, name=target[0])
    else:
        for f, t in zip(field, target):
            if t not in base._data:
                base = base.copyField(field=f, target=t)

    # write flags
    for t in target:
        base._flags[mask, t] = flag

    return base
