#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import logging
from saqc.core.register import register
from saqc.funcs.proc_functions import (
    proc_interpolateGrid,
    proc_shift,
    proc_fork,
    proc_resample,
    proc_projectFlags,
    proc_drop,
    proc_rename,
    ORIGINAL_SUFFIX,
)


logger = logging.getLogger("SaQC")

@register
def harm_shift2Grid(data, field, flagger, freq, method="nshift", to_drop=None, **kwargs):
    """
    A method to "regularize" data by shifting data points forward/backward to a regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Method keywords:

    'nshift' -  every grid point gets assigned the nearest value in its range ( range = +/-(freq/2) )
    'bshift' -  every grid point gets assigned its first succeeding value - if there is one available in the
            succeeding sampling interval. (equals resampling wih "first")
    'fshift'  -  every grid point gets assigned its ultimately preceeding value - if there is one available in
            the preceeding sampling interval. (equals resampling with "last")

    Note: the flags associated with every datapoint will just get shifted with them.

    Note: if there is no valid data (exisiing and not-na) available in a sampling interval assigned to a regular
    timestamp by the selected method, nan gets assigned to this timestamp. The associated flag will be of value
    flagger.UNFLAGGED.

    Note: all data nans get excluded defaultly from shifting. If drop_flags is None - all BAD flagged values get
    excluded as well.

    Note: the method will likely and significantly alter values and shape of data[field]. The original data is kept
    in the data dios and assigned to the fieldname field + "_original".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        The frequency of the grid you want to shift your data to.
    method : {'nshift', 'bshift', 'fshift'}, default 'nshift'
        Specifies if datapoints get propagated forwards, backwards or to the nearest grid timestamp.
        See description above for details
    drop_flags : {List[str], str}, default None
        Flagtypes you want to drop before shifting - effectively excluding values that are flagged
        with a flag in drop_flags from the shifting process. Default - results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_shift(
        data, field, flagger, freq, method, to_drop=to_drop, empty_intervals_flag=flagger.UNFLAGGED, **kwargs
    )
    return data, flagger


@register
def harm_aggregate2Grid(
    data, field, flagger, freq, value_func, flag_func=np.nanmax, method="nagg", to_drop=None, **kwargs
):
    """
    A method to "regularize" data by aggregating (resampling) data at a regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    The data will therefor get aggregated with a function, specified by the `value_func` parameter and
    the result gets projected onto the new timestamps with a method, specified by "method".

    The following method (keywords) are available:

    'nagg'  (aggreagtion to nearest) - all values in the range (+/- freq/2) of a grid point get aggregated with agg_func
            and assigned to it.
            Flags get aggregated by `flag_func` and assigned the same way.
    'bagg'  (backwards aggregation) - all values in a sampling interval get aggregated with agg_func and the result gets
            assigned to the last regular timestamp.
            Flags get aggregated by flag_func and assigned the same way.
    'fagg'  (forward aggregation) - all values in a sampling interval get aggregated with agg_func and the result gets
            assigned to the next regular timestamp.
            Flags get aggregated by flag_func and assigned the same way.


    Note, that, if there is no valid data (exisitng and not-na) available in a sampling interval assigned to a regular timestamp by the selected method,
    nan gets assigned to this timestamp. The associated flag will be of value flagger.UNFLAGGED.

    Note: the method will likely and significantly alter values and shape of data[field]. The original data is kept
    in the data dios and assigned to the fieldname field + "_original".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger
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
    drop_flags : {List[str], str}, default None
        Flagtypes you want to drop before aggregation - effectively excluding values that are flagged
        with a flag in drop_flags from the aggregation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_resample(
        data,
        field,
        flagger,
        freq,
        agg_func=value_func,
        flag_agg_func=flag_func,
        method=method,
        empty_intervals_flag=flagger.UNFLAGGED,
        to_drop=to_drop,
        all_na_2_empty=True,
        **kwargs,
    )
    return data, flagger


@register
def harm_linear2Grid(data, field, flagger, freq, to_drop=None, **kwargs):
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    Note: the method will likely and significantly alter values and shape of data[field]. The original data is kept
    in the data dios and assigned to the fieldname field + "_original".

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    flagger.UNFLAGGED.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    drop_flags : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in drop_flags from the interpolation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(
        data, field, flagger, freq, "time", to_drop=to_drop, empty_intervals_flag=flagger.UNFLAGGED, **kwargs
    )
    return data, flagger


@register
def harm_interpolate2Grid(
    data, field, flagger, freq, method, order=1, to_drop=None, **kwargs,
):
    """
    A method to "regularize" data by interpolating the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.

    Note, that, to perform a timestamp aware, linear interpolation, you have to pass 'time' as method, and NOT 'linear'.

    Note: the method will likely and significantly alter values and shape of data[field]. The original data is kept
    in the data dios and assigned to the fieldname field + "_original".

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    flagger.UNFLAGGED.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    order : int, default 1
        If your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    drop_flags : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in drop_flags from the interpolation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(
        data,
        field,
        flagger,
        freq,
        method=method,
        inter_order=order,
        to_drop=to_drop,
        empty_intervals_flag=flagger.UNFLAGGED,
        **kwargs,
    )
    return data, flagger


@register
def harm_deharmonize(data, field, flagger, method, to_drop=None, **kwargs):
    """
    The Function function "undoes" regularization, by regaining the original data and projecting the
    flags calculated for the regularized data onto the original ones.

    Afterwards the regularized data is removed from the data dios and 'field' will be associated
    to the original data "again".

    Wherever the flags in the original data are "better" then the regularized flags projected on them,
    they get overridden with this regularized flags value.

    Which regularized flags are to be projected on which original flags, is controlled by the "method" parameters.

    Generally, if you regularized with the method 'X', you should pass the method 'inverse_X' to the deharmonization.
    If you regularized with an interpolation, the method 'inverse_interpolation' would be the appropriate choice.
    Also you should pass the same drop flags keyword.

    The deharm methods in detail:
    ("original_flags" are associated with the original data that is to be regained,
    "regularized_flags" are associated with the regularized data that is to be "deharmonized",
    "freq" refers to the regularized datas sampling frequencie)

    'inverse_nagg' - all original_flags within the range +/- freq/2 of a regularized_flag, get assigned this
        regularized flags value. (if regularized_flags > original_flag)
    'inverse_bagg' - all original_flags succeeding a regularized_flag within the range of "freq", get assigned this
        regularized flags value. (if regularized_flag > original_flag)
    'inverse_fagg' - all original_flags preceeding a regularized_flag within the range of "freq", get assigned this
        regularized flags value. (if regularized_flag > original_flag)

    'inverse_interpolation' - all original_flags within the range +/- freq of a regularized_flag, get assigned this
        regularized flags value (if regularized_flag > original_flag).

    'inverse_nshift' - That original_flag within the range +/- freq/2, that is nearest to a regularized_flag, gets the
        regularized flags value. (if regularized_flag > original_flag)
    'inverse_bshift' - That original_flag succeeding a source flag within the range freq, that is nearest to a
        regularized_flag, gets assigned this regularized flags value. (if regularized_flag > original_flag)
    'inverse_nshift' - That original_flag preceeding a regularized flag within the range freq, that is nearest to a
        regularized_flag, gets assigned this regularized flags value. (if source_flag > original_flag)

    Parameters
    ----------
     data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-deharmonized.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.freq
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
            'inverse_interpolation'}
        The method used for projection of regularized flags onto opriginal flags. See description above for more
        details.
    drop_flags : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in drop_flags from the interpolation process. Default results in flagger.BAD
        values being dropped initially.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    data, flagger = proc_projectFlags(
        data, str(field) + ORIGINAL_SUFFIX, flagger, method, source=field, to_drop=to_drop, **kwargs
    )
    data, flagger = proc_drop(data, field, flagger)
    data, flagger = proc_rename(data, str(field) + ORIGINAL_SUFFIX, flagger, field)
    return data, flagger
