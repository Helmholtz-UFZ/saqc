#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.core import register
from saqc.funcs.interpolation import _SUPPORTED_METHODS
from saqc.lib.tools import evalFreqStr, filterKwargs, getFreqDelta, isflagged
from saqc.lib.ts_operators import aggregate2Freq, shift2Freq

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
        field : str
            The fieldname of the column, holding the data-to-be-regularized.

        freq : str
            An offset string. The frequency of the grid you want to interpolate your data at.

        Returns
        -------
        saqc.SaQC
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
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs,
    ) -> "SaQC":
        """
        Shift data points and flags to a regular frequency grid.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-shifted.

        freq : str
            Offset string. Sampling rate of the target frequency.

        method : {'fshift', 'bshift', 'nshift'}, default 'nshift'
            Method to propagate values:

            * 'nshift' : shift grid points to the nearest time stamp in the range = +/- 0.5 * ``freq``
            * 'bshift' : shift grid points to the first succeeding time stamp (if any)
            * 'fshift' : shift grid points to the last preceeding time stamp (if any)

        freq_check : {None, 'check', 'auto'}, default None
            * ``None`` : do not validate the ``freq`` string.
            * 'check' : check ``freq`` against an frequency estimation, produces a warning in case of miss matches.
            * 'auto' : estimate frequency, `freq` is ignored.

        Returns
        -------
        saqc.SaQC
        """
        warnings.warn(
            f"""
            The method `shift` is deprecated and will be removed with version 2.6 of saqc.
            To achieve the same behavior please use:
            `qc.align(field={field}, freq={freq}. method={method})`
            """,
            DeprecationWarning,
        )
        freq = evalFreqStr(freq, freq_check, self._data[field].index)
        return self.align(field=field, freq=freq, method=method, **kwargs)

    @register(mask=["field"], demask=[], squeeze=[])
    def resample(
        self: "SaQC",
        field: str,
        freq: str,
        func: Callable[[pd.Series], pd.Series] = np.mean,
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        maxna: Optional[int] = None,
        maxna_group: Optional[int] = None,
        maxna_flags: Optional[int] = None,  # TODO: still a case ??
        maxna_group_flags: Optional[int] = None,
        flag_func: Callable[[pd.Series], float] = max,
        freq_check: Optional[Literal["check", "auto"]] = None,
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
        field : str
            The fieldname of the column, holding the data-to-be-resampled.

        freq : str
            Offset string. Sampling rate of the target frequency grid.

        func : Callable
            Aggregation function. See notes for performance considerations.

        method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
            Specifies which intervals to be aggregated for a certain timestamp. (preceding,
            succeeding or "surrounding" interval). See description above for more details.

        maxna : {None, int}, default None
            Maximum number of allowed ``NaN``s in a resampling interval. If exceeded, the
            entire interval is filled with ``NaN``.

        maxna_group : {None, int}, default None
            Same as `maxna` but for consecutive NaNs.

        maxna_flags : {None, int}, default None
            Same as `max_invalid`, only applying for the flags. The flag regarded
            as "invalid" value, is the one passed to empty_intervals_flag (
            default=``BAD``). Also this is the flag assigned to invalid/empty intervals.

        maxna_group_flags : {None, int}, default None
            Same as `maxna_flags`, only applying onto flags. The flag regarded as
            "invalid" value, is the one passed to empty_intervals_flag. Also this is the
            flag assigned to invalid/empty intervals.

        flag_func : Callable, default: max
            The function you want to aggregate the flags with. It should be capable of
            operating on the flags dtype (usually ordered categorical).

        freq_check : {None, 'check', 'auto'}, default None

            * ``None``: do not validate frequency-string passed to `freq`
            * ``'check'``: estimate frequency and log a warning if estimate miss matchs
                frequency string passed to 'freq', or if no uniform sampling rate could be
                estimated
            * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

        Returns
        -------
        saqc.SaQC
        """

        datcol = self._data[field]

        if datcol.empty:
            # see for #GL-374
            datcol = pd.Series(index=pd.DatetimeIndex([]), dtype=datcol.dtype)

        freq = evalFreqStr(freq, freq_check, datcol.index)

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
            agg_func=flag_func,
            fill_value=np.nan,
            max_invalid_total=maxna_flags,
            max_invalid_consec=maxna_group_flags,
        )

        history = self._flags.history[field].apply(
            index=datcol.index,
            func=aggregate2Freq,
            func_kws=kws,
        )

        self._data[field] = datcol
        self._flags.history[field] = history
        return self

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        handles_target=True,  # target is mandatory in func, so its allowed
    )
    def concatFlags(
        self: "SaQC",
        field: str,
        target: str,
        method: Literal[
            "inverse_fagg",
            "inverse_bagg",
            "inverse_nagg",
            "inverse_fshift",
            "inverse_bshift",
            "inverse_nshift",
            "inverse_interpolation",
            "match",
        ] = "match",
        freq: str | None = None,
        drop: bool = False,
        squeeze: bool = False,
        overwrite: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Append the flags/history of ``field`` to ``target``. If necessary the flags are
        projected to the ``target`` frequency grid.

        Note
        ----
        To undo or backtrack resampling, shifting or interpolation operations, use the
        associated inversion method (e.g. to undo a former interpolation use
        ``method="inverse_interpolation"``).

        Parameters
        ----------
        field : str
            Fieldname of flags history to append.

        target : str
            Field name of flags history to append to.

        method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift', 'match'}, default 'match'
            Method to project the flags of ``field`` the flags to ``target``:

           * 'inverse_nagg': project a flag of ``field`` to all timestamps of ``target`` within the range +/- ``freq``/2.
           * 'inverse_bagg': project a flag of ``field`` to all preceeding timestamps of ``target`` within the range ``freq``
           * 'inverse_fagg': project a flag of ``field`` to all succeeding timestamps of ``target`` within the range ``freq``
           * 'inverse_interpolation' - project a flag of ``field`` to all timestamps of ``target`` within the range +/- ``freq``
           * 'inverse_nshift' - project a flag of ``field`` to the neaerest timestamps in ``target`` within the range +/- ``freq``/2
           * 'inverse_bshift' - project a flag of ``field`` to nearest preceeding timestamps in ``target``
           * 'inverse_nshift' - project a flag of ``field`` to nearest succeeding timestamps in ``target``
           * 'match' - project a flag of ``field`` to all identical timestamps ``target``

        freq : str or None, default None
            Projection range. If ``None`` the sampling frequency of ``field`` is used.

        drop : bool, default False
            Remove ``field`` if ``True``

        squeeze : bool, default False
            Squueze the history into a single column if ``True``. Function specific flag information is lost.

        overwrite: bool, default False
            Overwrite existing flags if ``True``

        Returns
        -------
        saqc.SaQC
        """
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

        if method[-13:] == "interpolation":
            ignore = _getChunkBounds(target_datcol, flagscol, freq)
            func = _inverseInterpolation
            func_kws = dict(freq=freq, chunk_bounds=ignore, target=dummy)

        elif method[-3:] == "agg":
            projection_method = METHOD2ARGS[method][0]
            tolerance = METHOD2ARGS[method][1](freq)
            func = _inverseAggregation
            func_kws = dict(freq=tolerance, method=projection_method, target=dummy)

        elif method[-5:] == "shift":
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

        if squeeze:
            history = history.squeeze(raw=True)

            meta = {
                "func": f"concatFlags",
                "args": (field,),
                "kwargs": {
                    "target": target,
                    "method": method,
                    "freq": freq,
                    "drop": drop,
                    "squeeze": squeeze,
                    "overwrite": overwrite,
                    **kwargs,
                },
            }
            self._flags.history[target].append(history, meta)
        else:
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
