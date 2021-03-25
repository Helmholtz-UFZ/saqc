#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple, Optional, Union, Any, Sequence
from typing_extensions import Literal

import numpy as np
import logging

import pandas as pd

from dios import DictOfSeries

from saqc.constants import *
from saqc.core.register import register, _isflagged
from saqc.flagger.history import applyFunctionOnHistory
from saqc.flagger.flags import Flagger
from saqc.funcs.tools import copy, drop, rename
from saqc.funcs.interpolation import interpolateIndex, _SUPPORTED_METHODS
from saqc.lib.tools import evalFreqStr, getFreqDelta
from saqc.lib.ts_operators import shift2Freq, aggregate2Freq
from saqc.lib.rolling import customRoller

logger = logging.getLogger("SaQC")

METHOD2ARGS = {
    "inverse_fshift": ("backward", pd.Timedelta),
    "inverse_bshift": ("forward", pd.Timedelta),
    "inverse_nshift": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "inverse_fagg": ("bfill", pd.Timedelta),
    "inverse_bagg": ("ffill", pd.Timedelta),
    "inverse_nagg": ("nearest", lambda x: pd.Timedelta(x) / 2),
    "match": (None, lambda _: "0min"),
}


@register(masking='none', module="resampling")
def aggregate(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        value_func,
        flag_func: Callable[[pd.Series], float] = np.nanmax,
        method: Literal["fagg", "bagg", "nagg"] = "nagg",
        flag: float = BAD,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    A method to "regularize" data by aggregating (resampling) data at a regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    The data will therefor get aggregated with a function, specified by the `value_func` parameter and
    the result gets projected onto the new timestamps with a method, specified by "method".

    The following method (keywords) are available:

    * ``'nagg'``: (aggreagtion to nearest) - all values in the range (+/- freq/2) of a grid point get aggregated with
      `agg_func`. and assigned to it. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'bagg'``: (backwards aggregation) - all values in a sampling interval get aggregated with agg_func and the
      result gets assigned to the last regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'fagg'``: (forward aggregation) - all values in a sampling interval get aggregated with agg_func and the result
      gets assigned to the next regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.

    Note, that, if there is no valid data (exisitng and not-na) available in a sampling interval assigned to a regular
    timestamp by the selected method, nan gets assigned to this timestamp. The associated flag will be of value
    ``UNFLAGGED``.

    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-regularized.

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.freq

    freq : str
        The sampling frequency the data is to be aggregated (resampled) at.

    value_func : Callable
        The function you want to use for aggregation.

    flag_func : Callable
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).

    method : {'fagg', 'bagg', 'nagg'}, default 'nagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceeding, succeeding or
        "surrounding" interval). See description above for more details.

    flag : float, default BAD
        flag to set.


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    return resample(
        data, field, flagger,
        freq=freq,
        agg_func=value_func,
        flag_agg_func=flag_func,
        method=method,
        flag=flag,
        **kwargs
    )


@register(masking='none', module="resampling")
def linear(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.

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

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.freq

    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    return interpolateIndex(data, field, flagger, freq, "time", **kwargs)


@register(masking='none', module="resampling")
def interpolate(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        method: _SUPPORTED_METHODS,
        order: int = 1,
        **kwargs,
) -> Tuple[DictOfSeries, Flagger]:
    """
    A method to "regularize" data by interpolating the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.

    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.

    Note: the `method` will likely and significantly alter values and shape of ``data[field]``. The original data is
    kept in the data dios and assigned to the fieldname ``field + '_original'``.

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

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.freq

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
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = copy(data, field, flagger, field + '_original')
    return interpolateIndex(data, field, flagger, freq, method=method, inter_order=order, **kwargs)


@register(masking='none', module="resampling")
def mapToOriginal(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        method: Literal[
            "inverse_fagg", "inverse_bagg", "inverse_nagg",
            "inverse_fshift", "inverse_bshift", "inverse_nshift",
            "inverse_interpolation"
        ],
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The Function function "undoes" regularization, by regaining the original data and projecting the
    flags calculated for the regularized data onto the original ones.

    Afterwards the regularized data is removed from the data dios and ``'field'`` will be associated
    with the original data "again".

    Wherever the flags in the original data are "better" then the regularized flags projected on them,
    they get overridden with this regularized flags value.

    Which regularized flags are to be projected on which original flags, is controlled by the "method" parameters.

    Generally, if you regularized with the method "X", you should pass the method "inverse_X" to the deharmonization.
    If you regularized with an interpolation, the method "inverse_interpolation" would be the appropriate choice.
    Also you should pass the same drop flags keyword.

    The deharm methods in detail:
    ("original_flags" are associated with the original data that is to be regained,
    "regularized_flags" are associated with the regularized data that is to be "deharmonized",
    "freq" refers to the regularized datas sampling frequencie)

    * ``'inverse_nagg'``: all original_flags within the range *+/- freq/2* of a regularized_flag, get assigned this
      regularized flags value. (if regularized_flags > original_flag)
    * ``'inverse_bagg'``: all original_flags succeeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_fagg'``: all original_flags preceeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)

    * ``'inverse_interpolation'``: all original_flags within the range *+/- freq* of a regularized_flag, get assigned this
      regularized flags value (if regularized_flag > original_flag).

    * ``'inverse_nshift'``: That original_flag within the range +/- *freq/2*, that is nearest to a regularized_flag,
      gets the regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_bshift'``: That original_flag succeeding a source flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_nshift'``: That original_flag preceeding a regularized flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if source_flag > original_flag)

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-deharmonized.

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.freq

    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
            'inverse_interpolation'}
        The method used for projection of regularized flags onto original flags. See description above for more
        details.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    newfield = str(field) + '_original'
    data, flagger = reindexFlags(data, newfield, flagger, method, source=field, to_mask=False)
    data, flagger = drop(data, field, flagger)
    return rename(data, newfield, flagger, field)


@register(masking='none', module="resampling")
def shift(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        freq_check: Optional[Literal["check", "auto"]] = None,  # TODO: not a user decision
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function to shift data and flags to a regular (equidistant) timestamp grid, according to ``method``.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-shifted.

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.

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
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    data, flagger = copy(data, field, flagger, field + '_original')
    return _shift(data, field, flagger, freq, method=method, freq_check=freq_check, **kwargs)


def _shift(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function to shift data points to regular (equidistant) timestamps.

    See Also
    --------
    shift : Main caller, docstring
    """
    flagged = _isflagged(flagger[field], kwargs['to_mask'])
    datcol = data[field]
    datcol[flagged] = np.nan
    freq = evalFreqStr(freq, freq_check, datcol.index)

    # do the shift
    datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)

    # do the shift on the history
    history = flagger.history[field]
    history.hist = shift2Freq(history.hist, method, freq, fill_value=UNTOUCHED)
    history.mask = shift2Freq(history.mask, method, freq, fill_value=False)

    # The last 2 lines left the history in an unstable state, Also we want to
    # append a dummy column, that represent the 'shift' in the history.
    # Luckily the append also fix the unstable state - noice.
    dummy = pd.Series(UNTOUCHED, index=datcol.index, dtype=float)
    history.append(dummy, force=True)

    flagger.history[field] = history
    data[field] = datcol
    return data, flagger


@register(masking='none', module="resampling")
def resample(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        agg_func: Callable[[pd.Series], pd.Series] = np.mean,
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        max_invalid_total_d: Optional[int] = None,
        max_invalid_consec_d: Optional[int] = None,
        max_invalid_consec_f: Optional[int] = None,
        max_invalid_total_f: Optional[int] = None,
        flag_agg_func: Callable[[pd.Series], float] = max,
        freq_check: Optional[Literal["check", "auto"]] = None,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function to resample the data. Afterwards the data will be sampled at regular (equidistant) timestamps
    (or Grid points). Sampling intervals therefor get aggregated with a function, specifyed by 'agg_func' parameter and
    the result gets projected onto the new timestamps with a method, specified by "method". The following method
    (keywords) are available:

    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the last
      grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the next
      grid point.


    Note, that. if possible, functions passed to agg_func will get projected internally onto pandas.resample methods,
    wich results in some reasonable performance boost - however, for this to work, you should pass functions that have
    the __name__ attribute initialised and the according methods name assigned to it.
    Furthermore, you shouldnt pass numpys nan-functions
    (``nansum``, ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and they will thus not
    trigger ``resample.func()``, but the slower ``resample.apply(nanfunc)``. Also, internally, no nans get passed to
    the functions anyway, so that there is no point in passing the nan functions.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the data-to-be-resampled.

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.

    freq : str
        An Offset String, that will be interpreted as the frequency you want to resample your data with.

    agg_func : Callable
        The function you want to use for aggregation.

    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding, succeeding or
        "surrounding" interval). See description above for more details.

    max_invalid_total_d : {None, int}, default None
        Maximum number of invalid (nan) datapoints, allowed per resampling interval. If max_invalid_total_d is
        exceeded, the interval gets resampled to nan. By default (``np.inf``), there is no bound to the number of nan
        values in an interval and only intervals containing ONLY nan values or those, containing no values at all,
        get projected onto nan

    max_invalid_consec_d : {None, int}, default None
        Maximum number of consecutive invalid (nan) data points, allowed per resampling interval.
        If max_invalid_consec_d is exceeded, the interval gets resampled to nan. By default (np.inf),
        there is no bound to the number of consecutive nan values in an interval and only intervals
        containing ONLY nan values, or those containing no values at all, get projected onto nan.

    max_invalid_total_f : {None, int}, default None
        Same as `max_invalid_total_d`, only applying for the flags. The flag regarded as "invalid" value,
        is the one passed to empty_intervals_flag (default=``BAD``).
        Also this is the flag assigned to invalid/empty intervals.

    max_invalid_consec_f : {None, int}, default None
        Same as `max_invalid_total_f`, only applying onto flags. The flag regarded as "invalid" value, is the one passed
        to empty_intervals_flag. Also this is the flag assigned to invalid/empty intervals.

    flag_agg_func : Callable, default: max
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).

    freq_check : {None, 'check', 'auto'}, default None

        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    flagged = _isflagged(flagger[field], kwargs['to_mask'])
    datcol = data[field]
    datcol[flagged] = np.nan
    freq = evalFreqStr(freq, freq_check, datcol.index)

    datcol = aggregate2Freq(
        datcol,
        method,
        freq,
        agg_func,
        fill_value=np.nan,
        max_invalid_total=max_invalid_total_d,
        max_invalid_consec=max_invalid_consec_d,
    )

    kws = dict(
        method=method,
        freq=freq,
        agg_func=flag_agg_func,
        fill_value=UNTOUCHED,
        max_invalid_total=max_invalid_total_f,
        max_invalid_consec=max_invalid_consec_f,
    )

    flagger.history[field] = applyFunctionOnHistory(
        flagger.history[field],
        hist_func=aggregate2Freq, hist_kws=kws,
        mask_func=aggregate2Freq, mask_kws=kws,
        last_column='dummy'
    )

    data[field] = datcol
    return data, flagger


def _getChunkBounds(target: pd.Series, flagscol: pd.Series, freq: str):
    chunk_end = target.reindex(flagscol.index, method='bfill', tolerance=freq)
    chunk_start = target.reindex(flagscol.index, method='ffill', tolerance=freq)
    ignore_flags = (chunk_end.isna() | chunk_start.isna())
    return ignore_flags


def _inverseInterpolation(source: pd.Series, target: pd.Series, freq: str, chunk_bounds) -> pd.Series:
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


def _inverseShift(source: pd.Series, target: pd.Series, drop_mask: pd.Series,
                  freq: str, method: str, fill_value) -> pd.Series:
    dtype = source.dtype

    target_drops = target[drop_mask]
    target = target[~drop_mask]
    flags_merged = pd.merge_asof(
        source,
        target.index.to_series(name='pre_index'),
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


@register(masking='none', module="resampling")
def reindexFlags(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        method: Literal[
            "inverse_fagg", "inverse_bagg", "inverse_nagg",
            "inverse_fshift", "inverse_bshift", "inverse_nshift"
        ],
        source: str,
        freq: Optional[str] = None,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The Function projects flags of "source" onto flags of "field". Wherever the "field" flags are "better" then the
    source flags projected on them, they get overridden with this associated source flag value.

    Which "field"-flags are to be projected on which source flags, is controlled by the "method" and "freq"
    parameters.

    method: (field_flag in associated with "field", source_flags associated with "source")

    'inverse_nagg' - all field_flags within the range +/- freq/2 of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    'inverse_bagg' - all field_flags succeeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    'inverse_fagg' - all field_flags preceeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)

    'inverse_interpolation' - all field_flags within the range +/- freq of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)

    'inverse_nshift' - That field_flag within the range +/- freq/2, that is nearest to a source_flag, gets the source
        flags value. (if source_flag > field_flag)
    'inverse_bshift' - That field_flag succeeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    'inverse_nshift' - That field_flag preceeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)

    'match' - any field_flag with a timestamp matching a source_flags timestamp gets this source_flags value
    (if source_flag > field_flag)

    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the data column, you want to project the source-flags onto.

    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.

    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift'}
        The method used for projection of source flags onto field flags. See description above for more details.

    source : str
        The source source of flags projection.

    freq : {None, str},default None
        The freq determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of source is used.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    flagscol = flagger[source]

    if freq is None:
        freq = getFreqDelta(flagscol.index)
        if freq is None and not method == 'match':
            raise ValueError('To project irregularly sampled data, either use method="match", or pass custom '
                             'projection range to freq parameter')

    target_datcol = data[field]
    target_flagscol = flagger[field]
    dummy = pd.Series(np.nan, target_flagscol.index, dtype=float)

    if method[-13:] == "interpolation":
        ignore = _getChunkBounds(target_datcol, flagscol, freq)
        func = _inverseInterpolation
        func_kws = dict(freq=freq, chunk_bounds=ignore, target=dummy)
        mask_kws = {**func_kws, 'chunk_bounds': []}

    elif method[-3:] == "agg" or method == "match":
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        func = _inverseAggregation
        func_kws = dict(freq=tolerance, method=projection_method, target=dummy)
        mask_kws = func_kws

    elif method[-5:] == "shift":
        drop_mask = (target_datcol.isna() | _isflagged(target_flagscol, kwargs['to_mask']))
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        func = _inverseShift
        kws = dict(freq=tolerance, method=projection_method, drop_mask=drop_mask, target=dummy)
        func_kws = {**kws, 'fill_value': UNTOUCHED}
        mask_kws = {**kws, 'fill_value': False}

    else:
        raise ValueError(f"unknown method {method}")

    history = applyFunctionOnHistory(flagger.history[source], func, func_kws, func, mask_kws, last_column=dummy)
    flagger.history[field] = flagger.history[field].append(history, force=False)
    return data, flagger
