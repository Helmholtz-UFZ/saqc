#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from dios import DictOfSeries
from saqc.constants import BAD, FILTER_ALL, UNFLAGGED
from saqc.core.register import _isflagged, flagging, register

if TYPE_CHECKING:
    from saqc.core.core import SaQC


class FlagtoolsMixin:
    @flagging()
    def flagDummy(self: "SaQC", field: str, **kwargs) -> "SaQC":
        """
        Function does nothing but returning data and flags.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        Returns
        -------
        saqc.SaQC
        """
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def forceFlags(self: "SaQC", field: str, flag: float = BAD, **kwargs) -> "SaQC":
        """
        Set whole column to a flag value.

        Parameters
        ----------
        field : str
            columns name that holds the data

        flag : float, default BAD
            flag to set

        kwargs : dict
            unused

        Returns
        -------
        saqc.SaQC

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

        Parameters
        ----------
        field : str
            columns name that holds the data

        kwargs : dict
            unused

        Returns
        -------
        saqc.SaQC

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

        return self.forceFlags(field, flag=UNFLAGGED, **kwargs)

    @register(mask=[], demask=[], squeeze=["field"])
    def flagUnflagged(self: "SaQC", field: str, flag: float = BAD, **kwargs) -> "SaQC":
        """
        Function sets a flag at all unflagged positions.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        flag : float, default BAD
            flag value to set

        kwargs : Dict
            unused

        Returns
        -------
        saqc.SaQC

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the data is not relevant
        for processing.

        See Also
        --------
        clearFlags : set whole column to UNFLAGGED
        forceFlags : set whole column to a flag value
        """
        unflagged = self._flags[field].isna() | (self._flags[field] == UNFLAGGED)
        self._flags[unflagged, field] = flag
        return self

    @register(mask=["field"], demask=["field"], squeeze=["field"])
    def flagManual(
        self: "SaQC",
        field: str,
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries, list, np.ndarray],
        method: Literal[
            "left-open", "right-open", "closed", "plain", "ontime"
        ] = "left-open",
        mformat: Literal["start-end", "mflag"] = "start-end",
        mflag: Any = 1,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag data by given, "manually generated" data.

        The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
        The format of mdata can be an indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
        but also can be a plain list- or array-like.
        How indexed mdata is aligned to data is specified via the `method` parameter.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

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
        saqc.SaQC

        Examples
        --------
        An example for mdata

        .. doctest:: ExampleFlagManual

           >>> mdata = pd.Series([1, 0, 1], index=pd.to_datetime(['2000-02-01', '2000-03-01', '2000-05-01']))
           >>> mdata
           2000-02-01    1
           2000-03-01    0
           2000-05-01    1
           dtype: int64

        On *dayly* data, with the 'ontime' method, only the provided timestamps are used.
        Bear in mind that only exact timestamps apply, any offset will result in ignoring
        the timestamp.

        .. doctest:: ExampleFlagManual

           >>> data = pd.Series(0, index=pd.to_datetime(['2000-01-31', '2000-02-01', '2000-02-02', '2000-03-01', '2000-05-01']), name='daily_data')
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mdata', method='ontime')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02    False
           2000-03-01    False
           2000-05-01     True
           Name: daily_data, dtype: bool

        With the 'right-open' method, the mdata is forward fill:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mdata', method='right-open')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02     True
           2000-03-01    False
           2000-05-01     True
           Name: daily_data, dtype: bool

        With the 'left-open' method, backward filling is used:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('daily_data', mdata, mflag=1, mformat='mdata', method='left-open')
           >>> qc.flags['daily_data'] > UNFLAGGED
           2000-01-31    False
           2000-02-01     True
           2000-02-02     True
           2000-03-01     True
           2000-05-01     True
           Name: daily_data, dtype: bool
        """
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

        Parameters
        ----------

        field : str or List of str
           Variable or list of variables, the flags of which are to be transferred.

        target : str or List of str
            Variable or list of variables, the flags of `field` are to be transferred to.

        Returns
        -------
        saqc.SaQC

        See Also
        --------
        * :py:meth:`saqc.SaQC.flagGeneric`
        * :py:meth:`saqc.SaQC.concatFlags`

        Examples
        --------
        First, generate some data with some flags:

        .. doctest:: exampleTransfer

           >>> data = pd.DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [1, 2]})
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagRange('a', max=1.5)
           >>> qc.flags.to_df()
           columns      a    b    c
           0         -inf -inf -inf
           1        255.0 -inf -inf

        Now we can project the flag from `a` to `b` via

        .. doctest:: exampleTransfer

           >>> qc = qc.transferFlags('a', target='b')
           >>> qc.flags.to_df()
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
           >>> qc.flags.to_df()
           columns      a      b      c
           0         -inf   -inf   -inf
           1        255.0  255.0  255.0
        """

        return self.concatFlags(field, target=target, method="match", squeeze=False)

    @flagging()
    def propagateFlags(
        self: "SaQC",
        field: str,
        window: Union[str, int],
        method: Literal["ffill", "bfill"] = "ffill",
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Flag values before or after flags set by the last test.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        window : int, str
            Size of the repetition window. An integer defines the exact number of repetitions,
            strings are interpreted as time offsets to fill with .

        method : {"ffill", "bfill"}
            Direction of repetetion. With "ffill" the subsequent values receive the flag to
            repeat, with "bfill" the previous values.

        flag : float, default BAD
            Flag to set.

        dfilter : float, default FILTER_ALL
            Threshold flag.

        Returns
        -------
        saqc.SaQC

        Examples
        --------
        First, generate some data and some flags:

        .. doctest:: propagateFlags

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
           Name: a, dtype: float64

        Now, to repeat the flag '255.0' two times in direction of ascending indices, execute:

        .. doctest:: propagateFlags

           >>> qc.propagateFlags('a', window=2, method="ffill").flags["a"]
           0     -inf
           1     -inf
           2     -inf
           3    255.0
           4    255.0
           5    255.0
           6     -inf
           Name: a, dtype: float64

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
           Name: a, dtype: float64

        If an explicit flag is passed, it will be used to fill the repetition window

        .. doctest:: propagateFlags

           >>> qc.propagateFlags('a', window=2, method="bfill", flag=111).flags["a"]
           0     -inf
           1    111.0
           2    111.0
           3    255.0
           4     -inf
           5     -inf
           6     -inf
           Name: a, dtype: float64
        """

        if method not in {"bfill", "ffill"}:
            raise ValueError(f"supported methods are 'bfill', 'ffill', got '{method}'")

        # get the last history column
        hc = self._flags.history[field].hist.iloc[:, -1].astype(float)

        if method == "bfill":
            hc = hc[::-1]

        # get dfilter from meta or get of rid of this and
        # consider everything != np.nan as flag
        flagged = _isflagged(hc, dfilter)

        repeated = (
            flagged.rolling(window, min_periods=1, closed="left")
            .max()
            .fillna(0)
            .astype(bool)
        )

        if method == "bfill":
            repeated = repeated[::-1]

        self._flags[repeated, field] = flag

        return self
