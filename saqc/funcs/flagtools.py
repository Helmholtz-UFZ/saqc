#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import BAD, FILTER_ALL, UNFLAGGED
from saqc.core import DictOfSeries, flagging, register
from saqc.core.history import History
from saqc.lib.tools import (
    initializeTargets,
    isflagged,
    isunflagged,
    multivariateParameters,
    toSequence,
)
from saqc.lib.types import (
    ArrayLike,
    Int,
    NewSaQCFields,
    OffsetStr,
    SaQC,
    SaQCFields,
    ValidatePublicMembers,
)


class FlagtoolsMixin(ValidatePublicMembers):
    @flagging()
    def flagDummy(self: SaQC, field: str, **kwargs) -> SaQC:
        """
        Function does nothing but returning data and flags.

        Parameters
        ----------

        """
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def forceFlags(self: SaQC, field: SaQCFields, flag: float = BAD, **kwargs) -> SaQC:
        """
        Set whole column to a flag value.

        Parameters
        ----------

        See also
        --------
        clearFlags : set whole column to UNFLAGGED
        flagUnflagged : set flag value at all unflagged positions

        """
        self._flags[:, field] = flag
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def clearFlags(self: SaQC, field: SaQCFields, **kwargs) -> SaQC:
        """
        Assign the flag UNFLAGGED to all timestamps.

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the data
        is not relevant for processing.
        A warning is triggered if the ``flag`` keyword is given, because
        the flags are always set to ``UNFLAGGED``.

        See Also
        --------
        forceFlags : Assign a specific flag value to all timestamps.
        flagUnflagged : Assign a specific flag value to all timestamps that have no flag value assigned.
        """
        # NOTE: do we really need this?
        if "flag" in kwargs:
            kwargs = {**kwargs}  # copy
            flag = kwargs.pop("flag")
            warnings.warn(f"`flag={flag}` is ignored here.")

        return self.forceFlags(field, flag=UNFLAGGED, **kwargs)

    @register(mask=[], demask=[], squeeze=["field"])
    def flagUnflagged(
        self: SaQC, field: SaQCFields, flag: float = BAD, **kwargs
    ) -> SaQC:
        """
        Assign a `flag` to all timestamps.

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the
        data is not relevant for processing.

        See also
        --------
        clearFlags : assign UNFLAGGED to all timestamps
        forceFlags : Assign a specific flag value to all timestamps.
        """
        unflagged = self._flags[field].isna() | (self._flags[field] == UNFLAGGED)
        self._flags[unflagged, field] = flag
        return self

    @flagging()
    def setFlags(
        self,
        field: SaQCFields,
        data: str | list | ArrayLike | pd.Series,
        override: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Include flags listed in external data.

        Parameters
        ----------

        data :
            Determines which timestamps to set flags at, depending on the passed type:

            * 1-d `array` or `List` of timestamps or `pandas.Index`: flag `field` with `flag` at every timestamp in `f_data`
            * 2-d `array` or List of tuples: for all elements `t[k]` out of f_data:
              flag `field` with `flag` at every timestamp in between `t[k][0]` and `t[k][1]`
            * pd.Series: flag `field` with `flag` in between any index and data value of the passed series
            * str: use the variable timeseries `f_data` as flagging template
            * pd.Series: flag `field` with `flag` in between any index and data value of the passed series
            * 1-d `array` or `List` of timestamps: flag `field` with `flag` at every timestamp in `f_data`
            * 2-d `array` or List of tuples: for all elements `t[k]` out of f_data:
              flag `field` with `flag` at every timestamp in between `t[k][0]` and `t[k][1]`
        override :
            determines if flags shall be assigned although the value in question already has a flag assigned.
        """
        to_flag = pd.Series(False, index=self._data[field].index)

        # check if f_data is meant to denote timestamps:
        if (isinstance(data, (list, np.ndarray, pd.Index))) and not isinstance(
            data[0], (tuple, np.ndarray)
        ):
            set_idx = pd.DatetimeIndex(data).intersection(to_flag.index)
            to_flag[set_idx] = True
        else:  # f_data denotes intervals:
            if isinstance(data, (str, pd.Series)):
                if isinstance(data, str):
                    flags_data = self._data[data]
                else:
                    flags_data = data
                intervals = flags_data.items()
            else:
                intervals = data
            for s in intervals:
                to_flag[s[0] : s[1]] = True

        # elif isinstance(f_data, list):
        if not override:

            to_flag &= isunflagged(self._flags[field], thresh=kwargs["dfilter"])
            # to_flag &= (self._flags[field] < flag) & (
            #     self._flags[field] >= kwargs["dfilter"]
            # )
        self._flags[to_flag.values, field] = flag
        return self

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def flagManual(
        self: SaQC,
        field: str,
        mdata: str | pd.Series | ArrayLike | list | pd.DataFrame,
        method: Literal[
            "left-open", "right-open", "closed", "plain", "ontime"
        ] = "left-open",
        mformat: Literal["start-end", "mflag"] = "start-end",
        mflag: Any = 1,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Include flags listed in external data.

        The method allows to integrate pre-existing flagging information.

            .. deprecated:: 2.6.0
               Deprecated Function. See :py:meth:`~saqc.SaQC.setFlags`.

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
        warnings.warn(
            "`flagManual` is deprecated and will be removed in version 2.8 of saqc. "
            "Please use `setFlags` for similar functionality.",
            DeprecationWarning,
        )

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
            app_entry = pd.Series(mdata.iloc[-1], dat.index.shift(freq="1min")[-1:])
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
        mask = mask.reindex(dat.index, fill_value=False)  # .fillna(False)

        self._flags[mask, field] = flag
        return self

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        handles_target=True,  # function defines a target parameter, so it needs to handle it
        multivariate=True,
    )
    def transferFlags(
        self: SaQC,
        field: SaQCFields,
        target: SaQCFields | NewSaQCFields | None = None,
        squeeze: bool = False,
        overwrite: bool = False,
        **kwargs,
    ) -> SaQC:
        """
        Transfer flags from one field to another.

        Flags present at timestamps in the source field(s) are also assigned to that same timestamps in the target field(s).

        Optionally, flags already assigned to target, are being overridden or squashed together with the new assignment, into a single flags column.

        Parameters
        ----------
        squeeze : bool
            If True, compress the history into a single column, losing function-specific flag information.

        overwrite : bool
            If True, existing flags in the target field are overwritten.

        Examples
        --------
        First, generate some data with flags:

        .. doctest:: exampleTransfer

           >>> import saqc
           >>> data = pd.DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [1, 2]})
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagRange('a', max=1.5)
           >>> qc.flags.to_pandas()
                  a    b    c
           0   -inf -inf -inf
           1  255.0 -inf -inf

        Project the flag from `a` to `b`:

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags('a', target='b')
           >>> qc.flags.to_pandas()
                  a      b    c
           0   -inf   -inf -inf
           1  255.0  255.0 -inf

        Project flags of `a` to both `b` and `c`:

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags(['a','a'], ['b', 'c'], overwrite=True)
           >>> qc.flags.to_pandas()
                  a      b      c
           0   -inf   -inf   -inf
           1  255.0  255.0  255.0

        See also
        --------
        * :py:meth:`saqc.SaQC.flagGeneric`
        * :py:meth:`saqc.SaQC.concatFlags`
        """
        fields, targets, broadcasting = multivariateParameters(field, target)
        meta = {
            "func": "transferFlags",
            "args": (),
            "kwargs": {
                "field": field,
                "target": target,
                "squeeze": squeeze,
                "overwrite": overwrite,
                **kwargs,
            },
        }

        for field, target in zip(fields, targets):
            # initialize non existing targets
            if target not in self._data:
                self._data[target] = pd.Series(np.nan, index=self._data[field].index)
                self._flags._data[target] = History(self._data[target].index)
            if not self._data[field].index.equals(self._data[target].index):
                raise ValueError(
                    f"All Field and Target indices must match!\n"
                    f"Indices of {field} and {target} seem to be not congruent within the context of the given\n"
                    f"- fields: {fields}\n "
                    f"- and targets: {targets}"
                )
            history = self._flags.history[field].copy(deep=True)

            if overwrite is False:
                mask = isflagged(self._flags[target], thresh=kwargs["dfilter"])
                history._hist[mask] = np.nan

            if squeeze:
                # add squeezed flags
                flags = history.squeeze(raw=True)
                history = History(index=history.index).append(flags, meta)
            elif broadcasting is False:
                # add an empty flags
                flags = pd.Series(np.nan, index=history.index, dtype=float)
                history.append(flags, meta)
            # else:
            #    broadcasting -> multiple fields will be written to one target
            #    only add the fields' histories and add an empty column later

            self._flags.history[target].append(history)

        if broadcasting and not squeeze:
            # add one final history column
            # all targets are identical, if we broadcast fields -> target
            target = targets[0]
            history = self._flags.history[target]
            flags = pd.Series(np.nan, index=history.index, dtype=float)
            self._flags.history[target].append(flags, meta)

        return self

    @flagging()
    def propagateFlags(
        self: SaQC,
        field: str,
        window: OffsetStr | (Int > 0),
        method: Literal["ffill", "bfill"] = "ffill",
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> SaQC:
        """
        Propagate already assigned flags along the date axis.

        Extent and direction of propagation can be controlled through parameters ``window`` and ``method``.

        Parameters
        ----------
        window : int or str
            Size of the repetition window. An integer defines the exact number
            of periods to propagate, while a string is interpreted as a time
            offset.

        method : {"ffill", "bfill"}
            Direction of propagation:
            * ``ffill`` — propagate flag to subsequent values
            * ``bfill`` — propagate flag to preceding values

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

        Now, to repeat the flag '255.0' two times in the direction of ascending
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

        Choosing "bfill" will result in:

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
        repetition window:

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
        self: SaQC,
        field: SaQCFields,
        group: Sequence[SaQC] | None = None,
        target: SaQCFields | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Logical AND operation for Flags.

        Flag the variable(s) `field` at every period, at wich `field` in all of the saqc objects in
        `group` is flagged.

        See Examples section for examples.

        Parameters
        ----------
        group:
            A collection of ``SaQC`` objects. Flag checks are performed on all ``SaQC`` objects
            based on the variables specified in ``field``. Whenever all monitored variables
            are flagged, the associated timestamps will receive a flag.

        Examples
        --------
        Flag data, if the values are above a certain threshold (determined by :py:meth:`~saqc.SaQC.flagRange`) AND if the values are
        constant for 3 periods (determined by :py:meth:`~saqc.SaQC.flagConstants`)

        .. doctest:: andGroupExample

           >>> dat = pd.Series([1,0,0,0,1,2,3,4,5,5,5,4], name='data', index=pd.date_range('2000', freq='10min', periods=12))
           >>> qc = saqc.SaQC(dat)
           >>> qc = qc.andGroup('data', group=[qc.flagRange('data', max=4), qc.flagConstants('data', thresh=0, window=3)])
           >>> qc.flags['data']
           2000-01-01 00:00:00     -inf
           2000-01-01 00:10:00     -inf
           2000-01-01 00:20:00     -inf
           2000-01-01 00:30:00     -inf
           2000-01-01 00:40:00     -inf
           2000-01-01 00:50:00     -inf
           2000-01-01 01:00:00     -inf
           2000-01-01 01:10:00     -inf
           2000-01-01 01:20:00    255.0
           2000-01-01 01:30:00    255.0
           2000-01-01 01:40:00    255.0
           2000-01-01 01:50:00     -inf
           Freq: 10min, dtype: float64

        Masking data, so that a test result only gets assigned during daytime (between 6 and 18 o clock for example).
        The daytime condition is generated via :py:meth:`~saqc.SaQC.flagGeneric`:

        .. doctest:: andGroupExample

           >>> from saqc.lib.tools import periodicMask
               >>> mask_func = lambda x: ~periodicMask(x.index, '06:00:00', '18:00:00', True)
           >>> dat = pd.Series(range(100), name='data', index=pd.date_range('2000', freq='4h', periods=100))
           >>> qc = saqc.SaQC(dat)
           >>> qc = qc.andGroup('data', group=[qc.flagRange('data', max=5), qc.flagGeneric('data', func=mask_func)])
           >>> qc.flags['data'].head(20)
           2000-01-01 00:00:00     -inf
           2000-01-01 04:00:00     -inf
           2000-01-01 08:00:00     -inf
           2000-01-01 12:00:00     -inf
           2000-01-01 16:00:00     -inf
           2000-01-01 20:00:00     -inf
           2000-01-02 00:00:00     -inf
           2000-01-02 04:00:00     -inf
           2000-01-02 08:00:00    255.0
           2000-01-02 12:00:00    255.0
           2000-01-02 16:00:00    255.0
           2000-01-02 20:00:00     -inf
           2000-01-03 00:00:00     -inf
           2000-01-03 04:00:00     -inf
           2000-01-03 08:00:00    255.0
           2000-01-03 12:00:00    255.0
           2000-01-03 16:00:00    255.0
           2000-01-03 20:00:00     -inf
           2000-01-04 00:00:00     -inf
           2000-01-04 04:00:00     -inf
           Freq: 4h, dtype: float64
        """
        return _groupOperation(
            saqc=self,
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
        self: SaQC,
        field: SaQCFields,
        group: Sequence[SaQC] | None = None,
        target: SaQCFields | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Logical OR operation for Flags.

        Flag the variable(s) `field` at every period, at wich `field` is flagged in at least one of the saqc objects
        in `group`.

        See Examples section for examples.

        Parameters
        ----------
        group:
            A collection of ``SaQC`` objects. Flag checks are performed on all ``SaQC`` objects
            based on the variables specified in :py:attr:`field`. Whenever any of monitored variables
            is flagged, the associated timestamps will receive a flag.

        Examples
        --------
        Flag data, if the values are above a certain threshold (determined by :py:meth:`~saqc.SaQC.flagRange`) OR if the values are
        constant for 3 periods (determined by :py:meth:`~saqc.SaQC.flagConstants`)

        .. doctest:: orGroupExample

           >>> dat = pd.Series([1,0,0,0,0,2,3,4,5,5,7,8], name='data', index=pd.date_range('2000', freq='10min', periods=12))
           >>> qc = saqc.SaQC(dat)
           >>> qc = qc.orGroup('data', group=[qc.flagRange('data', max=5), qc.flagConstants('data', thresh=0, window=3)])
           >>> qc.flags['data']
           2000-01-01 00:00:00     -inf
           2000-01-01 00:10:00    255.0
           2000-01-01 00:20:00    255.0
           2000-01-01 00:30:00    255.0
           2000-01-01 00:40:00    255.0
           2000-01-01 00:50:00     -inf
           2000-01-01 01:00:00     -inf
           2000-01-01 01:10:00     -inf
           2000-01-01 01:20:00     -inf
           2000-01-01 01:30:00     -inf
           2000-01-01 01:40:00    255.0
           2000-01-01 01:50:00    255.0
           Freq: 10min, dtype: float64

        """
        return _groupOperation(
            saqc=self,
            field=field,
            target=target,
            func=operator.or_,
            group=group,
            flag=flag,
            **kwargs,
        )


def _groupOperation(
    saqc: SaQC,
    field: SaQCFields,
    func: Callable[[pd.Series, pd.Series], pd.Series],
    group: Sequence[SaQC] | None = None,
    target: SaQCFields | None = None,
    flag: float = BAD,
    **kwargs,
) -> SaQC:
    """
    Perform a group operation on a collection of ``SaQC`` objects.

    This function applies a specified function to perform a group operation on a collection
    of `SaQC` objects. The operation involves checking specified :py:attr:`field` for flags,
    and if satisfied, assigning a flag value to corresponding timestamps.

    Parameters
    ----------
    saqc :
        The main `SaQC` object on which the output flags will be set.
    field :
        The field(s) to be checked for flags for all mebers of :py:attr:`group`.
    func :
        The function used to combine flags across the specified :py:attr:`field`
        and :py:attr:`group`.
    group :
        A sequence of ``SaQC`` objects forming the group for the group operation.
        If not provided, the operation is performed on the main ``SaQC`` object.

    Raises
    ------
    ValueError
        If input lengths or conditions are invalid.

    Notes
    -----
    - The `func` parameter should be a function that takes two boolean ``pd.Series`` objects,
      representing information on existing flags, and return a boolean ``pd.Series`` that
      representing the result od the elementwise logical combination of both.
    """

    def _flatten(seq: Sequence[str | Sequence[str]]) -> list[str]:
        out = []
        for e in seq:
            if isinstance(e, str):
                out.append(e)
            else:  # Sequence[str]
                out.extend(e)
        return out

    if target is None:
        target = field

    if isinstance(group, dict):
        warnings.warn(
            "The option to pass dictionaries to 'group' is deprecated and will be removed in version 2.7",
            DeprecationWarning,
        )
        group = list(group.keys())
        fields = list(group.values())

    fields = toSequence(field)
    targets = toSequence(target)

    if group is None or not group:
        group = [saqc]

    fields_ = fields[:]
    if len(fields_) == 1:
        # to simplify the retrieval from all groups...
        fields_ = fields * len(group)

    if len(fields_) != len(group):
        raise ValueError(
            "'field' needs to be a string or a sequence of the same length as 'group'"
        )

    # generate mask
    mask = pd.Series(dtype=bool)
    dfilter = kwargs.get("dfilter", FILTER_ALL)
    for qc, flds in zip(group, fields_):
        if set(flds := toSequence(flds)) - qc._flags.keys():
            raise KeyError(
                f"Failed to find one or more of the given variable(s), got {field}"
            )
        for f in flds:
            flagged = isflagged(qc._flags[f], thresh=dfilter)
            if mask.empty:
                mask = flagged
            mask = func(mask, flagged)

    targets = _flatten(targets)
    saqc = initializeTargets(saqc, _flatten(fields), targets, mask.index)

    # write flags
    for t in targets:
        saqc._flags[mask & isunflagged(saqc._flags[t], thresh=dfilter), t] = flag

    return saqc
