#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple, Optional, Union
from typing_extensions import Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import register, Flags
from saqc.core.register import _isflagged, processing
from saqc.lib.tools import evalFreqStr, getFreqDelta, filterKwargs
from saqc.lib.ts_operators import shift2Freq, aggregate2Freq
from saqc.funcs.interpolation import interpolateIndex, _SUPPORTED_METHODS
import saqc.funcs.tools as tools


METHOD2ARGS = {
    "inverse_fshift": ("backward", pd.Timedelta),
    "inverse_bshift": ("forward", pd.Timedelta),
    "inverse_nshift": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "inverse_fagg": ("bfill", pd.Timedelta),
    "inverse_bagg": ("ffill", pd.Timedelta),
    "inverse_nagg": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "match": (None, lambda _: "0min"),
}


@register(mask=["field"], demask=[], squeeze=[])
def linear(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: str,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-regularized.

    flags : saqc.Flags
        Container to store flags of the data.  freq

    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """
    reserved = ["method", "order", "limit", "downgrade"]
    kwargs = filterKwargs(kwargs, reserved)
    return interpolateIndex(data, field, flags, freq, "time", **kwargs)


@register(mask=["field"], demask=[], squeeze=[])
def interpolate(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: str,
    method: _SUPPORTED_METHODS,
    order: int = 1,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    A method to "regularize" data by interpolating the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.

    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``UNFLAGGED``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-regularized.

    flags : saqc.Flags
        Container to store flags of the data.  freq

    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method you want to apply.

    order : int, default 1
        If your selected interpolation method can be performed at different *orders* - here you pass the desired
        order.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """
    reserved = ["limit", "downgrade"]
    kwargs = filterKwargs(kwargs, reserved)
    return interpolateIndex(
        data, field, flags, freq, method=method, order=order, **kwargs
    )


@register(mask=["field"], demask=[], squeeze=[])
def shift(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: str,
    method: Literal["fshift", "bshift", "nshift"] = "nshift",
    freq_check: Optional[Literal["check", "auto"]] = None,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Function to shift data and flags to a regular (equidistant) timestamp grid, according to ``method``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-shifted.

    flags : saqc.Flags
        Container to store flags of the data.

    freq : str
        An frequency Offset String that will be interpreted as the sampling rate you want the data to be shifted to.

    method : {'fshift', 'bshift', 'nshift'}, default 'nshift'
        Specifies how misaligned data-points get propagated to a grid timestamp.
        Following choices are available:

        * 'nshift' : every grid point gets assigned the nearest value in its range. (range = +/- 0.5 * `freq`)
        * 'bshift' : every grid point gets assigned its first succeeding value, if one is available in
          the succeeding sampling interval.
        * 'fshift' : every grid point gets assigned its ultimately preceding value, if one is available in
          the preceeding sampling interval.

    freq_check : {None, 'check', 'auto'}, default None

        * ``None`` : do not validate frequency-string passed to `freq`
        * 'check' : estimate frequency and log a warning if estimate miss matches frequency string passed to `freq`,
          or if no uniform sampling rate could be estimated
        * 'auto' : estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """
    datcol = data[field]
    if datcol.empty:
        return data, flags

    freq = evalFreqStr(freq, freq_check, datcol.index)

    # do the shift
    datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)

    # do the shift on the history
    kws = dict(method=method, freq=freq)

    history = flags.history[field].apply(
        index=datcol.index,
        func_handle_df=True,
        func=shift2Freq,
        func_kws={**kws, "fill_value": np.nan},
    )

    flags.history[field] = history
    data[field] = datcol
    return data, flags


@register(mask=["field"], demask=[], squeeze=[])
def resample(
    data: DictOfSeries,
    field: str,
    flags: Flags,
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
) -> Tuple[DictOfSeries, Flags]:
    """
    Function to resample the data.

    The data will be sampled at regular (equidistant) timestamps aka. Grid points.
    Sampling intervals therefore get aggregated with a function, specified by
    'agg_func' parameter and the result gets projected onto the new timestamps with a
    method, specified by "method". The following method (keywords) are available:

    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get
        aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and
        the result gets assigned to the last grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and
        the result gets assigned to the next grid point.


    Note, that. if possible, functions passed to agg_func will get projected
    internally onto pandas.resample methods, wich results in some reasonable
    performance boost - however, for this to work, you should pass functions that
    have the __name__ attribute initialised and the according methods name assigned
    to it. Furthermore, you shouldnt pass numpys nan-functions (``nansum``,
    ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and
    they will thus not trigger ``resample.func()``, but the slower ``resample.apply(
    nanfunc)``. Also, internally, no nans get passed to the functions anyway,
    so that there is no point in passing the nan functions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-resampled.

    flags : saqc.Flags
        Container to store flags of the data.

    freq : str
        An Offset String, that will be interpreted as the frequency you want to
        resample your data with.

    func : Callable
        The function you want to use for aggregation.

    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding,
        succeeding or "surrounding" interval). See description above for more details.

    maxna : {None, int}, default None
        Maximum number NaNs in a resampling interval. If maxna is exceeded, the interval
        is set entirely to NaN.

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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """

    datcol = data[field]
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

    history = flags.history[field].apply(
        index=datcol.index,
        func=aggregate2Freq,
        func_kws=kws,
    )

    data[field] = datcol
    flags.history[field] = history
    return data, flags


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


@register(
    mask=[],
    demask=[],
    squeeze=[],
    handles_target=True,  # target is mandatory in func, so its allowed
)
def concatFlags(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    method: Literal[
        "inverse_fagg",
        "inverse_bagg",
        "inverse_nagg",
        "inverse_fshift",
        "inverse_bshift",
        "inverse_nshift",
        "inverse_interpolation",
    ],
    freq: Optional[str] = None,
    drop: Optional[bool] = False,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The Function appends flags history of ``fields`` to flags history of ``target``.
    Before Appending, columns in ``field`` history are projected onto the target index via ``method``

    method: (field_flag in associated with "field", source_flags associated with "source")

    'inverse_nagg' - all target_flags within the range +/- freq/2 of a field_flag, get assigned this field flags value.
        (if field_flag > target_flag)
    'inverse_bagg' - all target_flags succeeding a field_flag within the range of "freq", get assigned this field flags
        value. (if field_flag > target_flag)
    'inverse_fagg' - all target_flags preceeding a field_flag within the range of "freq", get assigned this field flags
        value. (if field_flag > target_flag)

    'inverse_interpolation' - all target_flags within the range +/- freq of a field_flag, get assigned this source flags value.
        (if field_flag > target_flag)

    'inverse_nshift' - That target_flag within the range +/- freq/2, that is nearest to a field_flag, gets the source
        flags value. (if field_flag > target_flag)
    'inverse_bshift' - That target_flag succeeding a field flag within the range freq, that is nearest to a
        field_flag, gets assigned this field flags value. (if field_flag > target_flag)
    'inverse_nshift' - That target_flag preceeding a field flag within the range freq, that is nearest to a
        field_flag, gets assigned this field flags value. (if field_flag > target_flag)

    'match' - any target_flag with a timestamp matching a field_flags timestamp gets this field_flags value
    (if field_flag > target_flag)

    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        Fieldname of flags history to append.

    flags : saqc.Flags
        Container to store flags of the data.

    target : str
        Field name of flags history to append to.

    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
             'match'}
        The method used for projection of ``field`` flags onto ``target`` flags. See description above for more details.

    freq : {None, str},default None
        The ``freq`` determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of ``field`` is used.

    drop : default False
        If set to `True`, the `field` column will be removed after processing

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """
    flagscol = flags[field]
    target_datcol = data[target]
    target_flagscol = flags[target]

    if target_datcol.empty or flagscol.empty:
        return data, flags

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
        drop_mask = target_datcol.isna() | _isflagged(
            target_flagscol, kwargs["dfilter"]
        )
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        func = _inverseShift
        kws = dict(
            freq=tolerance, method=projection_method, drop_mask=drop_mask, target=dummy
        )
        func_kws = {**kws, "fill_value": np.nan}

    elif method == "match":
        func = lambda x: x
        func_kws = {}

    else:
        raise ValueError(f"unknown method {method}")

    history = flags.history[field].apply(dummy.index, func, func_kws)
    flags.history[target].append(history)

    if drop:
        data, flags = tools.dropField(data=data, flags=flags, field=field)

    return data, flags
