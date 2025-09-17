#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import typing
import uuid
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc.constants
from saqc.core import History, register
from saqc.lib.docs import DOC_TEMPLATES
from saqc.lib.tools import getFreqDelta
from saqc.lib.ts_operators import isValid
from saqc.lib.types import (
    AGG_FUNC_LITERALS,
    METHOD_LITERALS,
    FreqStr,
    Int,
    OffsetLike,
    OffsetStr,
    SaQC,
    ValidatePublicMembers,
)

AGGFUNCS = list(typing.get_args(AGG_FUNC_LITERALS))

METHODINVERTS = {
    "fshift": "bshift",
    "bshift": "fshift",
    "nshift": "nshift",
    "fagg": "nagg",
    "bagg": "fagg",
    "nagg": "nagg",
    "nroll": "nroll",
    "froll": "broll",
    "broll": "froll",
    "mshift": "sshift",
    "sshift": "mshift",
    "match": "match",
}

METHOD2ARGS = {
    "inverse_fshift": "bshift",
    "inverse_bshift": "fshift",
    "inverse_nshift": "nshift",
    "inverse_fagg": "nagg",
    "inverse_bagg": "fagg",
    "inverse_nagg": "nagg",
    "inverse_interpolation": "sshift",
    "inverse_match": "match",
}

SHIFTMETHODS = ["fshift", "bshift", "nshift"]
PILLARMETHODS = ["mshift", "sshift"]


def _reindexerFromPillarPoints(
    method, data_aggregation, flags_aggregation, tolerance, datcol, index, idx_source
):

    if method == "mshift":
        na_wrapper = lambda x: x
    else:
        na_wrapper = lambda x, y=datcol.notna(): x.where(y, np.nan)

    grouper = pd.Series(np.arange(len(index)), index=index)
    grouper_b = grouper.reindex(idx_source, method="ffill", tolerance=tolerance)
    grouper_f = grouper.reindex(idx_source, method="bfill", tolerance=tolerance)

    if isinstance(flags_aggregation, str):
        flags_reindexer = lambda x: getattr(
            pd.concat(
                [
                    _reindexer(na_wrapper(x), "first", index, grouper_b, np.nan),
                    _reindexer(na_wrapper(x), "last", index, grouper_f, np.nan),
                ],
                axis=1,
            ),
            flags_aggregation,
        )(axis=1)

    elif pd.api.types.is_scalar(flags_aggregation):
        flags_reindexer = lambda x: pd.Series(flags_aggregation, index=index)
    else:
        raise NotImplementedError(
            "Pillarpoints reindexing not implemented for custom aggregation functions"
        )

    if method == "mshift":
        dummy = pd.Series(True, index=idx_source)
        dummy = (
            _reindexer(dummy, "first", index, grouper_b, np.nan).notna()
            & _reindexer(dummy, "last", index, grouper_f, np.nan).notna()
        )
        if data_aggregation in ["time", "linear"]:
            datcol = datcol.reindex(index.join(datcol.index, how="outer"))
            datcol = datcol.interpolate("time")
            datcol = datcol[index]
        elif pd.api.types.is_scalar(data_aggregation):
            datcol = pd.Series(data_aggregation, index=index)
        else:
            raise NotImplementedError(
                'Pillar reindexing not implemented for methods other than linear interpolation ("linear" / "time")'
            )
        new_dat = datcol.where(dummy, np.nan)

    elif method == "sshift":

        new_dat = pd.Series(data_aggregation, index=index)

    return flags_reindexer, new_dat


def _reindexerFromRoller(
    flags_reindexer,
    new_dat,
    datcol,
    flags_aggregation,
    data_aggregation,
    index,
    tolerance,
    method,
):
    if flags_reindexer is None:
        flags_reindexer = _constructRollingReindexer(
            flags_aggregation,
            index,
            tolerance,
            method,
        )
    if new_dat is None:
        new_dat = _constructRollingReindexer(
            data_aggregation, index, tolerance, method
        )(datcol)

    return flags_reindexer, new_dat


def _reindexerFromGrouper(
    flags_reindexer,
    new_dat,
    datcol,
    flags_aggregation,
    data_aggregation,
    index,
    grouper,
    bc_grouper,
):
    if flags_reindexer is None:
        flags_reindexer = _constructAggregationReindexer(
            flags_aggregation, index, grouper, np.nan, bc_grouper
        )
    if new_dat is None:
        new_dat = _constructAggregationReindexer(
            data_aggregation, index, grouper, np.nan, bc_grouper
        )(datcol)
    return flags_reindexer, new_dat


def _aggregationGrouper(method, index, idx_source, tolerance, datcol, broadcast):
    direction = method[0]
    rd = {"f": "bfill", "b": "ffill", "n": "nearest", "m": None}
    grouper = pd.Series(np.arange(len(index)), index=index)
    grouper = grouper.reindex(idx_source, method=rd[direction], tolerance=tolerance)

    if method == "nshift":
        # some extra acrobatics to find the projection-closest values in each group fro the nshift
        g_idx = grouper.dropna().astype(int)
        nearest = (
            pd.Series(abs(g_idx.index - index[g_idx.values]), index=g_idx.index)
            .groupby(g_idx.values)
            .idxmin()
        )
        grouper = grouper.loc[nearest.values]

    if broadcast:
        # to broadcast the aggregation one-to-many, we need a grouper from the inverted reindex
        bc_grouper = pd.Series(np.arange(len(datcol)), index=datcol.index)
        bc_grouper = bc_grouper.reindex(
            index, method=direction + rd[direction][1:], tolerance=tolerance
        )
    else:
        bc_grouper = None
    return grouper, bc_grouper


def _reindexer(col, func, idx, grouper, fill_val):
    if func in AGGFUNCS:
        group_aggregates = getattr(col.groupby(by=grouper), func)()
    else:
        # slower reindexer, but allows for custom functions not listed in AGGFUNCS
        group_aggregates = col.groupby(by=grouper).aggregate(func=func)
    out = pd.Series(fill_val, index=idx)
    out.iloc[group_aggregates.index.astype(int)] = group_aggregates.values.astype(float)
    return out


def _constructAggregationReindexer(func, idx, grouper, fill_val, bc_grouper=None):
    out_func = lambda x: _reindexer(x, func, idx, grouper, fill_val)
    if bc_grouper is not None:
        return lambda x: out_func(x).groupby(by=bc_grouper).transform(func)
    return out_func


def _rollingReindexer(x, idx, func, window, center, fwd):
    be_rolled = x.reindex(idx.join(x.index, how="outer"))
    if func in AGGFUNCS:
        be_rolled = getattr(be_rolled[::fwd].rolling(window, center=center), func)()[
            ::fwd
        ]
    else:
        be_rolled = be_rolled[::fwd].rolling(window, center=center).apply(func)[::fwd]
    return be_rolled[idx]


def _constructRollingReindexer(func, idx, window, direction):
    center = True if direction.startswith("n") else False
    fwd = -1 if direction.startswith("b") else 1
    return lambda x: _rollingReindexer(x, idx, func, window, center, fwd)


class ResamplingMixin(ValidatePublicMembers):
    def _invertLast(self, field):
        stack = []
        for meta in self._flags.history[field].meta:
            func = meta["func"]
            if func in ["reindex", "resample", "concatFlags"]:
                stack.append(meta["kwargs"].get("method"))
        if not stack:
            raise ValueError(
                "unable to derive an inversion method, please specify an appropiate 'method'"
            )

        done = False
        # remove already inverted methods from the meta iteratively, until None left, or the last not-inverted is
        # found
        while not done:
            rm = [
                k
                for k in range(len(stack) - 1)
                if stack[k + 1] == METHODINVERTS.get(stack[k], stack[k])
            ]
            if len(rm) > 0:
                stack = stack[: rm[0]] + stack[rm[0] + 2 :]
                if len(stack) == 0:
                    done = True
            else:
                done = True

        if len(stack) == 0:
            raise ValueError("Could not find no last reindexing to invert")

        reindex_method = METHODINVERTS.get(stack[-1], False)
        if reindex_method is False:
            raise ValueError(f"cant invert {stack[-1]}")
        return reindex_method

    def _retrieveIdx(self, idx, field):
        datcol = self.data[field]
        idx_source = datcol.index
        if not isinstance(idx, pd.Index):
            if idx in self._data.columns:
                idx = self.data[idx].index
            else:
                # frequency defined target index
                if len(datcol) == 0:
                    return pd.DatetimeIndex([]), idx_source, datcol
                idx_first = idx_source[0]
                idx_last = idx_source[-1]

                if isinstance(idx, str):
                    idx = pd.to_timedelta(idx)

                if isinstance(idx, pd.Timedelta):
                    idx = pd.date_range(
                        idx_first.floor(idx), idx_last.ceil(idx), freq=idx
                    )
                else:
                    raise ValueError(
                        f"cant make an index out of parameter 'idx's value: {idx}"
                    )
        return idx, idx_source, datcol

    @register(mask=["field"], demask=[], squeeze=[])
    def resample(
        self: SaQC,
        field: str,
        freq: FreqStr | pd.Timedelta,
        func: Callable[[pd.Series], pd.Series] | str = "mean",
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        maxna: (Int >= 0) | None = None,
        maxna_group: (Int >= 0) | None = None,
        squeeze: bool = False,
        **kwargs,
    ) -> SaQC:
        """
        Resample data points and flags to a regular frequency.

        The data will be sampled to regular (equidistant) timestamps.
        Sampling intervals therefore get aggregated with a function, specified by
        ``func``, the result is projected to the new timestamps using
        ``method``. The following methods are available:

        * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get
          aggregated with func and assigned to it.
        * ``'bagg'``: all values in a sampling interval get aggregated with func and
          the result gets assigned to the last grid point.
        * ``'fagg'``: all values in a sampling interval get aggregated with func and
          the result gets assigned to the next grid point.

        Note
        ----
        For perfomance reasons, ``func`` will be mapped to pandas.resample methods,
        if possible. However, for this to work, functions need an initialized
        ``__name__`` attribute, holding the function's name. Furthermore, you should
        not pass numpys nan-functions (``nansum``, ``nanmean``,...) because they
        cannot be optimised and the handling of ``NaN`` is already taken care of.

        Parameters
        ----------
        freq :
            Offset string. Sampling rate of the target frequency grid.

        func :
            Aggregation function. See notes for performance considerations.

        method :
            Specifies which intervals to be aggregated for a certain timestamp. (preceding,
            succeeding or "surrounding" interval). See description above for more details.

        maxna :
            Maximum number of allowed ``NaN``s in a resampling interval. If exceeded, the
            aggregation of the interval evaluates to ``NaN``.

        maxna_group :
            Same as `maxna` but for consecutive NaNs.
        """

        validator = lambda x: isValid(
            x, max_nan_total=maxna, max_nan_consec=maxna_group
        )
        tmp_val_field = str(uuid.uuid4())
        # parametrise reindexer for the interval validation:
        # we check any interval for sufficing the maxna and maxnagroup condition
        # broadcast is set to False, since we mimic "pandas resample"
        self = self.reindex(
            field,
            target=tmp_val_field,
            index=freq,
            method=method,
            data_aggregation=validator,
            broadcast=False,
            **kwargs,
        )
        # repeat the reindexing with the selected resampling func func
        self = self.reindex(
            field,
            index=freq,
            method=method,
            data_aggregation=func,
            squeeze=squeeze,
            overrride=True,
            broadcast=False,
            **kwargs,
        )
        # where the validation returned False, overwrite the resampling result:
        self = self.processGeneric(
            [field, tmp_val_field],
            target=field,
            func=lambda x, y: x.where(y.astype(bool), np.nan),
        )
        self = self.dropField(tmp_val_field)

        r_meta = {
            "field": field,
            "func": "resample",
            "args": (),
            "kwargs": {
                "freq": freq,
                "func": func,
                "method": method,
                "maxna": maxna,
                "maxna_group": maxna_group,
                **kwargs,
            },
        }
        self._flags.history[field] = self._flags.history[field][:, :-1]
        self._flags.history[field].meta = self._flags.history[field].meta[:-1] + [
            r_meta
        ]

        return self

    @register(
        mask=["field"],
        demask=[],
        squeeze=[],
        docstring={"target": DOC_TEMPLATES["target"]},
    )
    def reindex(
        self: SaQC,
        field: str,
        index: FreqStr | pd.DatetimeIndex | str,
        method: METHOD_LITERALS = "match",
        tolerance: OffsetStr | OffsetLike | None = None,
        data_aggregation: AGG_FUNC_LITERALS | Callable | float | None = None,
        flags_aggregation: AGG_FUNC_LITERALS | Callable | float | None = None,
        broadcast: bool = True,
        squeeze: bool = False,
        override: bool = False,
        **kwargs,
    ) -> SaQC:
        """
        Change a variables index.

        Simultaneously changes the indices of the data, flags and the history assigned to `field`.

        Parameters
        ----------

        index :
            Determines the new index.

            * If an `offset` string: new index will range from start to end of the original index of
              `field`, exhibting a uniform sampling rate of `idx`
            * If a `str` that matches a field present in the `SaQC` object, that fields index will be
              used as new index of `field`
            * If an `pd.index` object is passed, that will be the new index of `field`.

        method :
           Determines which of the origins indexes periods to comprise into the calculation of a new flag and a new data value at
           any period of the new index.

           * Aggregations Reindexer. Aggregations are data and flags independent, (pure) index selection methods:
           * `'bagg'`/`'fagg'`: "backwards/forwards aggregation". Any new index period gets assigned an
             aggregation of the values at periods in the original index, that lie between itself and its successor/predecessor.
           * `'nagg'`: "nearest aggregation": Any new index period gets assigned an aggregation of the values at periods
             in the original index between its direcet predecessor and successor, it is the nearest neighbor to.
           * Rolling reindexer. Rolling reindexers are equal to aggregations, when projecting between
             regular and irregular sampling grids forth and back. But due to there simple rolling window construction, they are
             easier to comprehend, predict and parametrize. On the downside, they are much more expensive computationally and
             Also, periods can get included in the aggregation to multpiple target periods, (when rolling windows overlap).
           * `'broll'`/`'froll'`: Any new index period gets assigned an aggregation of all the values at periods
             of the original index, that fall into a directly preceeding/succeeding window of size `reindex_window`.
           * Shifts. Shifting methods are shortcuts for aggregation reindex methods, combined with selecting
             'last' or 'first' as the `data_aggregation` method. Therefor, both, the `flags_aggregation`
             and the `data_aggregation` are ignored when using a `shift` reindexer. Also, periods
             where the data evaluates to `NaN` are dropped before shift index selection.
           * `'bshift'`/`fshift`: "backwards/forwards shift". Any new index period gets assigned the
             first/last valid (not a data NaN) value it succeeds/preceeds
           * `'nshift'`: "nearest shift": Any new index period gets assigned the value of its closest neighbor in the
             original index.
           * Pillar point Mappings. Index selection method designed to select indices suitable for
             linearly interpolating index values from surrounding pillar points in the original index, or inverting such
             a selection. Periods where the data evaluates to `NaN`, are dropped from consideration.
           * `'mshift'`: "Merge" predecessors and successors. Any new index period gets assigned an aggregation/interpolation comprising
             the last and the next valid period in the original index.
           * `'sshift'`: "Split"-map values onto predecessors and successors. Same as `mshift`, but with a correction that prevents missing value
             flags from being mapped to continuous data chunk bounds.
           * Inversion of last method: try to select the method, that
           * `'invert``

        tolerance :
           Limiting the distance, values can be shifted or comprised into aggregation.

        data_aggregation :
            Function string or custom Function, determining how to aggregate new data values
            from the values at the periods selected according to the `index_selection_method`.
            If a scalar value is passed, the new data series will just evaluate to that scalar at any new index.

        flags_aggregation :
            Function string or custom Function, determining how to aggregate new flags values
            from the values at the periods selected according to the `index_selection_method`.
            If a scalar value is passed, the new flags series will just evaluate to that scalar at any new index.

        broadcast :
            Weather to propagate aggregation result to full reindex window when using aggregation reindexer.
            (as opposed to only assign to next/previous/closest)

        Notes
        -----

        .. figure:: /resources/images/reindexMethods.png


        Examples
        --------

        Generate some example data with messed up 1 day-ish sampling rate

        .. doctest:: reindexExample

           >>> import pandas as pd
           >>> import saqc
           >>> import numpy as np
           >>> from saqc.constants import FILTER_NONE
           >>> np.random.seed(23)
           >>> index = pd.DatetimeIndex(pd.date_range('2000', freq='1d', periods=23))
           >>> index += pd.Index([pd.Timedelta(f'{k}min') for k in np.random.randint(-360,360,23)])
           >>> drops = np.random.randint(0,20,3)
           >>> drops.sort()
           >>> index=index[np.r_[0:drops[0],drops[0]+1:drops[1],drops[1]+1:drops[2],drops[2]+1:23]]
           >>> data = pd.Series(np.abs(np.arange(-10,10)), index=index, name='data')
           >>> data # doctest: +SKIP
           2000-01-01 03:55:00    10
           2000-01-03 02:08:00     9
           2000-01-03 18:31:00     8
           2000-01-04 21:57:00     7
           2000-01-06 01:40:00     6
           2000-01-06 23:47:00     5
           2000-01-09 04:02:00     4
           2000-01-10 05:05:00     3
           2000-01-10 18:06:00     2
           2000-01-12 01:09:00     1
           2000-01-13 02:44:00     0
           2000-01-13 18:49:00     1
           2000-01-15 05:46:00     2
           2000-01-16 01:39:00     3
           2000-01-17 05:49:00     4
           2000-01-17 21:12:00     5
           2000-01-18 18:12:00     6
           2000-01-21 03:20:00     7
           2000-01-21 22:57:00     8
           2000-01-23 03:51:00     9
           Name: data, dtype: int64

        Performing linear alignment to 2 days grid, with and without limiting the reindexing range:

        .. doctest:: reindexExample

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.reindex('data', target='linear', index='2D', method='mshift', data_aggregation='linear')
           >>> qc = qc.reindex('data', target='limited_linear', index='2D', method='mshift', data_aggregation='linear', tolerance='1D')
           >>> qc.data # doctest: +SKIP
                              data |               linear |       limited_linear |
           ======================= | ==================== | ==================== |
           2000-01-01 03:55:00  10 | 1999-12-31       NaN | 1999-12-31       NaN |
           2000-01-03 02:08:00   9 | 2000-01-02  9.565453 | 2000-01-02       NaN |
           2000-01-03 18:31:00   8 | 2000-01-04  7.800122 | 2000-01-04  7.800122 |
           2000-01-04 21:57:00   7 | 2000-01-06  6.060132 | 2000-01-06       NaN |
           2000-01-06 01:40:00   6 | 2000-01-08  4.536523 | 2000-01-08       NaN |
           2000-01-06 23:47:00   5 | 2000-01-10  3.202927 | 2000-01-10  3.202927 |
           2000-01-09 04:02:00   4 | 2000-01-12  1.037037 | 2000-01-12       NaN |
           2000-01-10 05:05:00   3 | 2000-01-14  1.148307 | 2000-01-14       NaN |
           2000-01-10 18:06:00   2 | 2000-01-16  2.917016 | 2000-01-16  2.917016 |
           2000-01-12 01:09:00   1 | 2000-01-18  5.133333 | 2000-01-18  5.133333 |
           2000-01-13 02:44:00   0 | 2000-01-20  6.521587 | 2000-01-20       NaN |
           2000-01-13 18:49:00   1 | 2000-01-22  8.036332 | 2000-01-22       NaN |
           2000-01-15 05:46:00   2 | 2000-01-24       NaN | 2000-01-24       NaN |
           2000-01-16 01:39:00   3 |                      |                      |
           2000-01-17 05:49:00   4 |                      |                      |
           2000-01-17 21:12:00   5 |                      |                      |
           2000-01-18 18:12:00   6 |                      |                      |
           2000-01-21 03:20:00   7 |                      |                      |
           2000-01-21 22:57:00   8 |                      |                      |
           2000-01-23 03:51:00   9 |                      |                      |
           <BLANKLINE>

        Setting a flag, reindexing the linearly aligned field with the originl index (deharmonisation")

        .. doctest:: reindexExample

           >>> qc = qc.setFlags('linear', data=['2000-01-16'])
           >>> qc = qc.reindex('linear', index='data', tolerance='2D', method='sshift', dfilter=FILTER_NONE)
           >>> qc.flags[['data', 'linear']] # doctest: +SKIP
                               data |                     linear |
           ======================== | ========================== |
           2000-01-01 03:55:00 -inf | 2000-01-01 03:55:00   -inf |
           2000-01-03 02:08:00 -inf | 2000-01-03 02:08:00   -inf |
           2000-01-03 18:31:00 -inf | 2000-01-03 18:31:00   -inf |
           2000-01-04 21:57:00 -inf | 2000-01-04 21:57:00   -inf |
           2000-01-06 01:40:00 -inf | 2000-01-06 01:40:00   -inf |
           2000-01-06 23:47:00 -inf | 2000-01-06 23:47:00   -inf |
           2000-01-09 04:02:00 -inf | 2000-01-09 04:02:00   -inf |
           2000-01-10 05:05:00 -inf | 2000-01-10 05:05:00   -inf |
           2000-01-10 18:06:00 -inf | 2000-01-10 18:06:00   -inf |
           2000-01-12 01:09:00 -inf | 2000-01-12 01:09:00   -inf |
           2000-01-13 02:44:00 -inf | 2000-01-13 02:44:00   -inf |
           2000-01-13 18:49:00 -inf | 2000-01-13 18:49:00   -inf |
           2000-01-15 05:46:00 -inf | 2000-01-15 05:46:00  255.0 |
           2000-01-16 01:39:00 -inf | 2000-01-16 01:39:00  255.0 |
           2000-01-17 05:49:00 -inf | 2000-01-17 05:49:00   -inf |
           2000-01-17 21:12:00 -inf | 2000-01-17 21:12:00   -inf |
           2000-01-18 18:12:00 -inf | 2000-01-18 18:12:00   -inf |
           2000-01-21 03:20:00 -inf | 2000-01-21 03:20:00   -inf |
           2000-01-21 22:57:00 -inf | 2000-01-21 22:57:00   -inf |
           2000-01-23 03:51:00 -inf | 2000-01-23 03:51:00   -inf |
           <BLANKLINE>

        Now, `linear` flags can easily be appended to data, to complete "deharm" step.

        Another example: Shifting to nearest regular frequeny and back. Note, how 'nearest' - style reindexers "invert" themselfs.

        .. doctest:: reindexExample

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.reindex('data', index='1D', target='n_shifted', method='nshift')
           >>> qc = qc.reindex('n_shifted', index='data', target='n_shifted_undone', method='nshift')
           >>> qc.data # doctest: +SKIP
                              data |        n_shifted |          n_shifted_undone |
           ======================= | ================ | ========================= |
           2000-01-01 03:55:00  10 | 2000-01-01  10.0 | 2000-01-01 03:55:00  10.0 |
           2000-01-03 02:08:00   9 | 2000-01-02   NaN | 2000-01-03 02:08:00   9.0 |
           2000-01-03 18:31:00   8 | 2000-01-03   9.0 | 2000-01-03 18:31:00   8.0 |
           2000-01-04 21:57:00   7 | 2000-01-04   8.0 | 2000-01-04 21:57:00   7.0 |
           2000-01-06 01:40:00   6 | 2000-01-05   7.0 | 2000-01-06 01:40:00   6.0 |
           2000-01-06 23:47:00   5 | 2000-01-06   6.0 | 2000-01-06 23:47:00   5.0 |
           2000-01-09 04:02:00   4 | 2000-01-07   5.0 | 2000-01-09 04:02:00   4.0 |
           2000-01-10 05:05:00   3 | 2000-01-08   NaN | 2000-01-10 05:05:00   3.0 |
           2000-01-10 18:06:00   2 | 2000-01-09   4.0 | 2000-01-10 18:06:00   2.0 |
           2000-01-12 01:09:00   1 | 2000-01-10   3.0 | 2000-01-12 01:09:00   1.0 |
           2000-01-13 02:44:00   0 | 2000-01-11   2.0 | 2000-01-13 02:44:00   0.0 |
           2000-01-13 18:49:00   1 | 2000-01-12   1.0 | 2000-01-13 18:49:00   1.0 |
           2000-01-15 05:46:00   2 | 2000-01-13   0.0 | 2000-01-15 05:46:00   2.0 |
           2000-01-16 01:39:00   3 | 2000-01-14   1.0 | 2000-01-16 01:39:00   3.0 |
           2000-01-17 05:49:00   4 | 2000-01-15   2.0 | 2000-01-17 05:49:00   4.0 |
           2000-01-17 21:12:00   5 | 2000-01-16   3.0 | 2000-01-17 21:12:00   5.0 |
           2000-01-18 18:12:00   6 | 2000-01-17   4.0 | 2000-01-18 18:12:00   6.0 |
           2000-01-21 03:20:00   7 | 2000-01-18   5.0 | 2000-01-21 03:20:00   7.0 |
           2000-01-21 22:57:00   8 | 2000-01-19   6.0 | 2000-01-21 22:57:00   8.0 |
           2000-01-23 03:51:00   9 | 2000-01-20   NaN | 2000-01-23 03:51:00   9.0 |
                                   | 2000-01-21   7.0 |                           |
                                   | 2000-01-22   8.0 |                           |
                                   | 2000-01-23   9.0 |                           |
                                   | 2000-01-24   NaN |                           |
           <BLANKLINE>

        Furthermoer, forward/backward style reindexers can be inverted by backward/forward style reindexers:

        .. doctest:: reindexExample

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.reindex('data', target='sum_aggregate', index='3D', method='fagg', data_aggregation='sum')
           >>> qc = qc.setFlags('sum_aggregate', data=['2000-01-18', '2000-01-24'])
           >>> qc = qc.reindex('sum_aggregate', target='bagg', index='data', method='bagg', dfilter=FILTER_NONE)
           >>> qc = qc.reindex('sum_aggregate', target='bagg_limited', index='data', method='bagg', tolerance='2D', dfilter=FILTER_NONE)
           >>> qc.flags # doctest: +SKIP
                               data |     sum_aggregate |                       bagg |               bagg_limited |
           ======================== | ================= | ========================== | ========================== |
           2000-01-01 03:55:00 -inf | 1999-12-31   -inf | 2000-01-01 03:55:00   -inf | 2000-01-01 03:55:00   -inf |
           2000-01-03 02:08:00 -inf | 2000-01-03   -inf | 2000-01-03 02:08:00   -inf | 2000-01-03 02:08:00   -inf |
           2000-01-03 18:31:00 -inf | 2000-01-06   -inf | 2000-01-03 18:31:00   -inf | 2000-01-03 18:31:00   -inf |
           2000-01-04 21:57:00 -inf | 2000-01-09   -inf | 2000-01-04 21:57:00   -inf | 2000-01-04 21:57:00   -inf |
           2000-01-06 01:40:00 -inf | 2000-01-12   -inf | 2000-01-06 01:40:00   -inf | 2000-01-06 01:40:00   -inf |
           2000-01-06 23:47:00 -inf | 2000-01-15   -inf | 2000-01-06 23:47:00   -inf | 2000-01-06 23:47:00   -inf |
           2000-01-09 04:02:00 -inf | 2000-01-18  255.0 | 2000-01-09 04:02:00   -inf | 2000-01-09 04:02:00   -inf |
           2000-01-10 05:05:00 -inf | 2000-01-21   -inf | 2000-01-10 05:05:00   -inf | 2000-01-10 05:05:00   -inf |
           2000-01-10 18:06:00 -inf | 2000-01-24  255.0 | 2000-01-10 18:06:00   -inf | 2000-01-10 18:06:00   -inf |
           2000-01-12 01:09:00 -inf |                   | 2000-01-12 01:09:00   -inf | 2000-01-12 01:09:00   -inf |
           2000-01-13 02:44:00 -inf |                   | 2000-01-13 02:44:00   -inf | 2000-01-13 02:44:00   -inf |
           2000-01-13 18:49:00 -inf |                   | 2000-01-13 18:49:00   -inf | 2000-01-13 18:49:00   -inf |
           2000-01-15 05:46:00 -inf |                   | 2000-01-15 05:46:00  255.0 | 2000-01-15 05:46:00   -inf |
           2000-01-16 01:39:00 -inf |                   | 2000-01-16 01:39:00  255.0 | 2000-01-16 01:39:00  255.0 |
           2000-01-17 05:49:00 -inf |                   | 2000-01-17 05:49:00  255.0 | 2000-01-17 05:49:00  255.0 |
           2000-01-17 21:12:00 -inf |                   | 2000-01-17 21:12:00  255.0 | 2000-01-17 21:12:00  255.0 |
           2000-01-18 18:12:00 -inf |                   | 2000-01-18 18:12:00   -inf | 2000-01-18 18:12:00   -inf |
           2000-01-21 03:20:00 -inf |                   | 2000-01-21 03:20:00  255.0 | 2000-01-21 03:20:00   -inf |
           2000-01-21 22:57:00 -inf |                   | 2000-01-21 22:57:00  255.0 | 2000-01-21 22:57:00   -inf |
           2000-01-23 03:51:00 -inf |                   | 2000-01-23 03:51:00  255.0 | 2000-01-23 03:51:00  255.0 |
           <BLANKLINE>

        """
        if method == "invert":
            method = self._invertLast(field)
            if method.endswith("agg"):
                broadcast = True

        data_aggregation = data_aggregation or np.nan
        flags_aggregation = flags_aggregation or "max"

        flags_reindexer = None
        new_dat = None

        if method.endswith("roll") and (tolerance is None):
            raise ValueError(
                'When using rolling indexer, parameter "tolerance" has to be assigned an extension '
                "in terms of an offset string. got None."
            )

        if method == "match":
            data_aggregation = lambda x: x

        if method == "mshift":
            data_aggregation = "linear"

        index, idx_source, datcol = self._retrieveIdx(index, field)

        if len(index) == 0:
            new_dat = pd.Series([], index=pd.DatetimeIndex([]))
            flags_reindexer = lambda x: pd.Series([], pd.DatetimeIndex([]))

        if method == "match":
            data_aggregation = lambda x: x

        if method in SHIFTMETHODS:
            broadcast = False
            data_aggregation = "last" if method == "fshift" else "first"
            flags_aggregation = data_aggregation
            datcol = datcol.dropna()
            idx_source = datcol.index
            if len(datcol) == 0:
                new_dat = pd.Series(np.nan, index=index)

        # assign source invariant reindexers
        if (
            (data_aggregation not in AGGFUNCS)
            and pd.api.types.is_scalar(data_aggregation)
            and (new_dat is None)
        ):
            new_dat = pd.Series(data_aggregation, index=index)

        if (
            (flags_aggregation not in AGGFUNCS)
            and pd.api.types.is_scalar(data_aggregation)
            and (flags_reindexer is None)
        ):
            flags_reindexer = lambda x: pd.Series(flags_aggregation, index=index)

        if method.endswith("agg") | (method in SHIFTMETHODS) | method.endswith("match"):

            grouper, bc_grouper = _aggregationGrouper(
                method, index, idx_source, tolerance, datcol, broadcast
            )
            flags_reindexer, new_dat = _reindexerFromGrouper(
                flags_reindexer,
                new_dat,
                datcol,
                flags_aggregation,
                data_aggregation,
                index,
                grouper,
                bc_grouper,
            )

        elif method.endswith("roll"):
            flags_reindexer, new_dat = _reindexerFromRoller(
                flags_reindexer,
                new_dat,
                datcol,
                flags_aggregation,
                data_aggregation,
                index,
                tolerance,
                method,
            )

        elif method in PILLARMETHODS:
            flags_reindexer, new_dat = _reindexerFromPillarPoints(
                method,
                data_aggregation,
                flags_aggregation,
                tolerance,
                datcol,
                index,
                idx_source,
            )
        else:
            raise ValueError(f"reindex method {method} unknown...")

        history = self._flags.history[field].apply(index, flags_reindexer, {})
        meta = {
            "func": "reindex",
            "args": (),
            "kwargs": {"field": field, "method": method, **kwargs},
        }
        if squeeze:
            flags = history.squeeze(raw=True)
            history = History(index=history.index).append(flags, meta)
        else:
            flags = pd.Series(UNFLAGGED if override else np.nan, index=history.index)
            history.append(flags, meta)

        self._flags.history[field] = history
        self._data[field] = new_dat
        return self

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        handles_target=True,  # target is mandatory in func, so its allowed
        docstring={"target": DOC_TEMPLATES["target"]},
    )
    def concatFlags(
        self: SaQC,
        field: str,
        target: str | None = None,
        method: Literal[
            "fagg",
            "bagg",
            "nagg",
            "fshift",
            "bshift",
            "nshift",
            "sshift",
            "mshift",
            "match",
            "auto",
            "linear",
            "pad",
        ] = "auto",
        invert: bool = True,
        freq: FreqStr | pd.Timedelta | None = None,
        drop: bool = False,
        squeeze: bool = False,
        override: bool = False,
        **kwargs,
    ) -> SaQC:
        """
        Project flags/history of :py:attr:`field` to :py:attr:`target` and adjust to the frequeny grid
        of :py:attr:`target` by 'undoing' former interpolation, shifting or resampling operations

        Note
        ----
        To just use the appropriate inversion with regard to a certain method, set the
        `invert` parameter to True and pass the method you want to invert.

        To backtrack a preveous resampling, shifting or interpolation operation automatically, set `method='auto'`

        Parameters
        ----------

        method :
            Method to project the flags of :py:attr:`field` to the flags to :py:attr:`target`:

           * ``'auto'``: invert the last alignment/resampling operation (that is not already inverted)
           * ``'nagg'``: project a flag of :py:attr:`field` to all timestamps of
             :py:attr:`target` within the range +/- :py:attr:`freq`/2.
           * ``'bagg'``: project a flag of :py:attr:`field` to all preceeding timestamps
             of :py:attr:`target` within the range :py:attr:`freq`
           * ``'fagg'``: project a flag of :py:attr:`field` to all succeeding timestamps
             of :py:attr:`target` within the range :py:attr:`freq`
           * ``'interpolation'`` - project a flag of :py:attr:`field` to all timestamps
             of :py:attr:`target` within the range +/- :py:attr:`freq`
           * ``'sshift'`` - same as interpolation
           * ``'nshift'`` - project a flag of :py:attr:`field` to the neaerest timestamps
             in :py:attr:`target` within the range +/- :py:attr:`freq`/2
           * ``'bshift'`` - project a flag of :py:attr:`field` to nearest preceeding
             timestamps in :py:attr:`target`
           * ``'nshift'`` - project a flag of :py:attr:`field` to nearest succeeding
             timestamps in :py:attr:`target`
           * ``'match'`` - project a flag of :py:attr:`field` to all identical timestamps
             :py:attr:`target`

        invert :
            If True, not the actual method is applied, but its inversion-method.

        freq :
            Projection range. If ``None`` the sampling frequency of :py:attr:`field` is used.

        drop :
            Remove :py:attr:`field` if ``True``

        squeeze :
            Squeeze the history into a single column if ``True``, function specific flag information is lost.

        override :
            Overwrite existing flags if ``True``
        """
        if method.split("_")[0] == "inverse":
            warnings.warn(
                f""" Referring to a method that would invert a method 'A` via 'inverse_A' is deprecated and will
                be removed in version 2.7. Please use method={method.split('_')[-1]} together
                with invert=True.
                """,
                DeprecationWarning,
            )
            method = method.split("_")[-1]
            invert = True

        if method == "match":
            warnings.warn(
                f"The method 'match' is deprecated and will be removed "
                f"in version 2.7 of SaQC. Please use `SaQC.transferFlags(field={field}, "
                f"target={target}, squeeze={squeeze}, overwrite={override})` instead",
                DeprecationWarning,
            )
            return self.transferFlags(
                field=field, target=target, squeeze=squeeze, overwrite=override
            )

        if target is None:
            target = field

        # map liases/traditional notions to correct method names (important to find auto-invert)
        method = "mshift" if method == "linear" else method
        method = "fshift" if method == "pad" else method

        # if last method is to be inverted automatically, the last method that is not already inverted
        # has to be found in the meta:
        if method == "auto":
            reindex_method = self._invertLast(field)
        else:
            reindex_method = METHODINVERTS[method] if invert else method

        freq = freq or getFreqDelta(self._data[field].index)
        temp_field = str(uuid.uuid4())
        # parametrise reindexer:
        # - FILTER_NONE is set, since we want to backproject ALL flags (not only the good ones)
        # - no data_aggregation is set, since we only are interested in the reindexed flags
        # - flags_aggregation defaults to max
        # - broadcast is True, since we project a flag onto all the periods its data value was aggregated
        #   from when aligning
        self = self.reindex(
            field,
            target=temp_field,
            index=self._data[target].index,
            tolerance=freq,
            dfilter=saqc.constants.FILTER_NONE,
            override=override,
            method=reindex_method,
            broadcast=True,
        )
        # transfer reindexed flags
        self = self.transferFlags(temp_field, target, squeeze=squeeze)
        self = self.dropField(field=temp_field)
        if drop:
            return self.dropField(field=field)

        return self
