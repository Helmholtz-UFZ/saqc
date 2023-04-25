#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.constants import UNFLAGGED
from saqc.core import register
from saqc.core.history import History
from saqc.lib.docs import DOC_TEMPLATES
from saqc.lib.tools import filterKwargs, getFreqDelta, isflagged
from saqc.lib.ts_operators import aggregate2Freq

if TYPE_CHECKING:
    from saqc import SaQC


METHOD2ARGS = {
    "inverse_fshift": ("backward", pd.Timedelta),
    "inverse_bshift": ("forward", pd.Timedelta),
    "inverse_nshift": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "inverse_fagg": ("bfill", pd.Timedelta),
    "inverse_bagg": ("ffill", pd.Timedelta),
    "inverse_nagg": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "match": (None, lambda _: "0min"),
}


class ResamplingMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def linear(
        self: "SaQC",
        field: str,
        freq: str,
        **kwargs,
    ) -> "SaQC":
        """
        A method to "regularize" data by interpolating linearly the data at regular timestamp.

        A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

        Interpolated values will get assigned the worst flag within freq-range.

        Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
        not-na) datapoint preceeding them and one succeeding them within freq range.
        Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
        ``UNFLAGGED``.

        Parameters
        ----------
        freq :
            An offset string. The frequency of the grid you want to interpolate your data at.
        """
        reserved = ["method", "order", "limit", "downgrade"]
        kwargs = filterKwargs(kwargs, reserved)
        return self.interpolateIndex(field, freq, "time", **kwargs)

    @register(mask=["field"], demask=[], squeeze=[])
    def shift(
        self: "SaQC",
        field: str,
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        **kwargs,
    ) -> "SaQC":
        """
        Shift data points and flags to a regular frequency grid.

        Parameters
        ----------
        freq :
            Offset string. Sampling rate of the target frequency.

        method :
            Method to propagate values:

            * 'nshift' : shift grid points to the nearest time stamp in the range = +/- 0.5 * ``freq``
            * 'bshift' : shift grid points to the first succeeding time stamp (if any)
            * 'fshift' : shift grid points to the last preceeding time stamp (if any)
        """
        warnings.warn(
            f"""
            The method `shift` is deprecated and will be removed with version 2.6 of saqc.
            To achieve the same behavior please use:
            `qc.align(field={field}, freq={freq}. method={method})`
            """,
            DeprecationWarning,
        )

        return self.align(field=field, freq=freq, method=method, **kwargs)

    @register(mask=["field"], demask=[], squeeze=[])
    def resample(
        self: "SaQC",
        field: str,
        freq: str,
        func: Callable[[pd.Series], pd.Series] = np.mean,
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        maxna: int | None = None,
        maxna_group: int | None = None,
        **kwargs,
    ) -> "SaQC":
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

        func : default mean
            Aggregation function. See notes for performance considerations.

        method :
            Specifies which intervals to be aggregated for a certain timestamp. (preceding,
            succeeding or "surrounding" interval). See description above for more details.

        maxna :
            Maximum number of allowed ``NaN``s in a resampling interval. If exceeded, the
            entire interval is filled with ``NaN``.

        maxna_group :
            Same as `maxna` but for consecutive NaNs.
        """

        datcol = self._data[field]

        if datcol.empty:
            # see for #GL-374
            datcol = pd.Series(index=pd.DatetimeIndex([]), dtype=datcol.dtype)

        datcol = aggregate2Freq(
            datcol,
            method,
            freq,
            func,
            fill_value=np.nan,
            max_invalid_total=maxna,
            max_invalid_consec=maxna_group,
        )

        kws = dict(
            method=method,
            freq=freq,
            agg_func=max,
            fill_value=np.nan,
            max_invalid_total=maxna,
            max_invalid_consec=maxna_group,
        )

        history = self._flags.history[field].apply(
            index=datcol.index,
            func=aggregate2Freq,
            func_kws=kws,
        )
        meta = {
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
        flagcol = pd.Series(UNFLAGGED, index=history.index)
        history.append(flagcol, meta)

        self._data[field] = datcol
        self._flags.history[field] = history
        return self

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        handles_target=True,  # target is mandatory in func, so its allowed
        docstring={"target": DOC_TEMPLATES["target"]},
    )
    def concatFlags(
        self: "SaQC",
        field: str,
        target: str | None = None,
        method: Literal[
            "inverse_fagg",
            "inverse_bagg",
            "inverse_nagg",
            "inverse_fshift",
            "inverse_bshift",
            "inverse_nshift",
            "inverse_interpolation",
            "match",
            "auto",
        ] = "match",
        freq: str | None = None,
        drop: bool = False,
        squeeze: bool = False,
        overwrite: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Project flags/history of :py:attr:`field` to :py:attr:`target` and adjust to the frequeny grid
        of :py:attr:`target` by 'undoing' former interpolation, shifting or resampling operations

        Note
        ----
        To undo or backtrack resampling, shifting or interpolation operations, use the
        associated inversion method (e.g. to undo a former interpolation use
        ``method="inverse_interpolation"``).

        Parameters
        ----------
        method :
            Method to project the flags of :py:attr:`field` the flags to :py:attr:`target`:

           * ``'auto'``: inverse the last alignment/resampling operations
           * ``'inverse_nagg'``: project a flag of :py:attr:`field` to all timestamps of
             :py:attr:`target` within the range +/- :py:attr:`freq`/2.
           * ``'inverse_bagg'``: project a flag of :py:attr:`field` to all preceeding timestamps
             of :py:attr:`target` within the range :py:attr:`freq`
           * ``'inverse_fagg'``: project a flag of :py:attr:`field` to all succeeding timestamps
             of :py:attr:`target` within the range :py:attr:`freq`
           * ``'inverse_interpolation'`` - project a flag of :py:attr:`field` to all timestamps
             of :py:attr:`target` within the range +/- :py:attr:`freq`
           * ``'inverse_nshift'`` - project a flag of :py:attr:`field` to the neaerest timestamps
             in :py:attr:`target` within the range +/- :py:attr:`freq`/2
           * ``'inverse_bshift'`` - project a flag of :py:attr:`field` to nearest preceeding
             timestamps in :py:attr:`target`
           * ``'inverse_nshift'`` - project a flag of :py:attr:`field` to nearest succeeding
             timestamps in :py:attr:`target`
           * ``'match'`` - project a flag of :py:attr:`field` to all identical timestamps
             :py:attr:`target`

        freq :
            Projection range. If ``None`` the sampling frequency of :py:attr:`field` is used.

        drop :
            Remove :py:attr:`field` if ``True``

        squeeze :
            Squueze the history into a single column if ``True``. Function specific flag information is lost.

        overwrite :
            Overwrite existing flags if ``True``
        """

        if target is None:
            target = field

        flagscol = self._flags[field]
        target_datcol = self._data[target]
        target_flagscol = self._flags[target]

        if target_datcol.empty or flagscol.empty:
            return self

        dummy = pd.Series(np.nan, target_flagscol.index, dtype=float)

        if freq is None:
            freq = getFreqDelta(flagscol.index)
            if freq is None and not method == "match":
                raise ValueError(
                    'To project irregularly sampled data, either use method="match", or '
                    "pass custom projection range to freq parameter."
                )

        if method == "auto":
            stack = []
            for meta in self._flags.history[field].meta:
                func = meta["func"]
                meth = meta["kwargs"].get("method")
                if func in ("align", "resample"):
                    if meth[1:] in ("agg", "shift"):
                        stack.append(f"inverse_{meth}")
                    else:
                        stack.append("inverse_interpolation")
                elif func == "concatFlags":
                    stack.pop()
            if not stack:
                raise ValueError(
                    "unable to derive an inversion method, please specify an appropiate 'method'"
                )
            method = stack[-1]

        if method.endswith("interpolation"):
            ignore = _getChunkBounds(target_datcol, flagscol, freq)
            func = _inverseInterpolation
            func_kws = dict(freq=freq, chunk_bounds=ignore, target=dummy)

        elif method.endswith("agg"):
            projection_method = METHOD2ARGS[method][0]
            tolerance = METHOD2ARGS[method][1](freq)
            func = _inverseAggregation
            func_kws = dict(freq=tolerance, method=projection_method, target=dummy)

        elif method.endswith("shift"):
            drop_mask = target_datcol.isna() | isflagged(
                target_flagscol, kwargs["dfilter"]
            )
            projection_method = METHOD2ARGS[method][0]
            tolerance = METHOD2ARGS[method][1](freq)
            func = _inverseShift
            kws = dict(
                freq=tolerance,
                method=projection_method,
                drop_mask=drop_mask,
                target=dummy,
            )
            func_kws = {**kws, "fill_value": np.nan}

        elif method == "match":
            func = lambda x: x
            func_kws = {}

        else:
            raise ValueError(f"unknown method {method}")

        history = self._flags.history[field].apply(dummy.index, func, func_kws)
        if overwrite is False:
            mask = isflagged(self._flags[target], thresh=kwargs["dfilter"])
            history._hist[mask] = np.nan

        # append a dummy column
        meta = {
            "func": f"concatFlags",
            "args": (),
            "kwargs": {
                "field": field,
                "target": target,
                "method": method,
                "freq": freq,
                "drop": drop,
                "squeeze": squeeze,
                "overwrite": overwrite,
                **kwargs,
            },
        }

        if squeeze:
            flags = history.squeeze(raw=True)
            history = History(index=history.index)
        else:
            flags = pd.Series(np.nan, index=history.index, dtype=float)

        history.append(flags, meta)
        self._flags.history[target].append(history)

        if drop:
            return self.dropField(field=field)

        return self


def _getChunkBounds(target: pd.Series, flagscol: pd.Series, freq: str):
    chunk_end = target.reindex(flagscol.index, method="bfill", tolerance=freq)
    chunk_start = target.reindex(flagscol.index, method="ffill", tolerance=freq)
    ignore_flags = chunk_end.isna() | chunk_start.isna()
    return ignore_flags


def _inverseInterpolation(
    source: pd.Series, target: pd.Series, freq: str, chunk_bounds
) -> pd.Series:
    source = source.copy()
    if len(chunk_bounds) > 0:
        source[chunk_bounds] = np.nan
    backprojected = source.reindex(target.index, method="bfill", tolerance=freq)
    fwrdprojected = source.reindex(target.index, method="ffill", tolerance=freq)
    return pd.concat([backprojected, fwrdprojected], axis=1).max(axis=1)


def _inverseAggregation(
    source: Union[pd.Series, pd.DataFrame],
    target: Union[pd.Series, pd.DataFrame],
    freq: str,
    method: str,
):
    return source.reindex(target.index, method=method, tolerance=freq)


def _inverseShift(
    source: pd.Series,
    target: pd.Series,
    drop_mask: pd.Series,
    freq: str,
    method: str,
    fill_value,
) -> pd.Series:
    dtype = source.dtype

    target_drops = target[drop_mask]
    target = target[~drop_mask]
    flags_merged = pd.merge_asof(
        source,
        target.index.to_series(name="pre_index"),
        left_index=True,
        right_index=True,
        tolerance=freq,
        direction=method,
    )
    flags_merged.dropna(subset=["pre_index"], inplace=True)
    flags_merged = flags_merged.set_index(["pre_index"]).squeeze()
    target[flags_merged.index] = flags_merged.values

    # reinsert drops
    source = target.reindex(target.index.union(target_drops.index))
    source.loc[target_drops.index] = target_drops.values

    return source.fillna(fill_value).astype(dtype, copy=False)
