#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import dios


from saqc.lib.ts_operators import interpolateNANs, aggregate2Freq, shift2Freq
from saqc.core.register import register
from saqc.lib.tools import toSequence



logger = logging.getLogger("SaQC")


class Heap:
    INDEX = "initial_ts"
    DATA = "original_data"
    FLAGGER = "original_flagger"
    FREQ = "freq"
    METHOD = "reshape_method"
    DROP = "drop_flags"


HARM_2_DEHARM = {
    "fshift": "invert_fshift",
    "bshift": "invert_bshift",
    "nshift": "invert_nearest",
    "fagg": "invert_fshift",
    "bagg": "invert_bshift",
    "nagg": "invert_nearest",
    "fagg_no_deharm": "regain",
    "bagg_no_deharm": "regain",
    "nagg_no_deharm": "regain",
}


def harmWrapper(heap={}):
    # NOTE:
    # (1) - harmonization will ALWAYS flag flagger.BAD all the np.nan values and afterwards DROP ALL
    #       flagger.BAD flagged values from flags frame for further flagging!!!!!!!!!!!!!!!!!!!!!
    def harmonize(
        data,
        field,
        flagger,
        freq,
        inter_method,
        reshape_method,
        inter_agg=np.nanmean,
        inter_order=1,
        inter_downcast=False,
        reshape_agg=np.nanmax,
        reshape_missing_flag=None,
        reshape_shift_comment=False,
        drop_flags=None,
        data_missing_value=np.nan,
        **kwargs,
    ):
        data = data.copy()

        # get data of variable
        flagger_merged = flagger.slice(field=field)
        dat_col = data[field]

        # now we send the flags frame in its current shape to the future:
        heap[field] = {
            Heap.DATA: dat_col,
            Heap.FLAGGER: flagger_merged,
            Heap.FREQ: freq,
            Heap.METHOD: reshape_method,
            Heap.DROP: drop_flags,
        }

        # now we can manipulate it without loosing information gathered before harmonization
        dat_col, flagger_merged_clean, _ = _outsortCrap(dat_col, field, flagger_merged, drop_flags=drop_flags)

        # interpolation! (yeah)
        dat_col, chunk_bounds = _interpolateGrid(
            dat_col,
            freq,
            method=inter_method,
            order=inter_order,
            agg_method=inter_agg,
            downcast_interpolation=inter_downcast,
        )

        # flags now have to be carefully adjusted according to the changes/shifts we did to data
        flagger_merged_clean_reshaped = _reshapeFlags(
            flagger_merged_clean,
            field,
            freq=dat_col.index.freq,
            method=reshape_method,
            agg_method=reshape_agg,
            missing_flag=reshape_missing_flag,
            set_shift_comment=reshape_shift_comment,
            block_flags=chunk_bounds,
            **kwargs,
        )

        flagger_out = flagger.getFlagger(drop=field).setFlagger(flagger_merged_clean_reshaped)
        data[field] = dat_col
        return data, flagger_out


    def deharmonize(data, field, flagger, co_flagging=False, **kwargs):

        # Check if there is backtracking information available for actual harmonization resolving
        if field not in heap:
            logger.warning(
                'No backtracking data for resolving harmonization of "{}". Reverse projection of flags gets'
                " skipped!".format(field)
            )
            return data, flagger

        # get some deharm configuration infos from the heap:
        harm_info = heap.pop(field)
        resolve_method = HARM_2_DEHARM[harm_info[Heap.METHOD]]

        # retrieve data and flags from the merged saqc-conform data frame (and by that get rid of blow-up entries).
        flagger_harmony = flagger.slice(field=field)
        dat_col = data[field]

        # reconstruct the drops that were performed before harmonization
        _, flagger_original_clean, drop_mask = _outsortCrap(
            dat_col, field, harm_info[Heap.FLAGGER], drop_flags=harm_info[Heap.DROP]
        )

        # with reconstructed pre-harmonization flags-frame -> perform the projection of the flags calculated for
        # the harmonized timeseries, onto the original timestamps
        flagger_back = _backtrackFlags(
            flagger_harmony,
            flagger_original_clean,
            harm_info[Heap.FLAGGER],
            harm_info[Heap.FREQ],
            track_method=resolve_method,
            co_flagging=co_flagging,
        )

        flags_col = flagger_back.getFlags(field)

        dat_col = harm_info[Heap.DATA].reindex(flags_col.index, fill_value=np.nan)
        dat_col.name = field

        # bye bye data
        flagger_out = flagger.getFlagger(drop=field).setFlagger(flagger_back)
        data[field] = dat_col

        assert (data[field].index == flagger_out.getFlags(field).index).all()
        return data, flagger_out

    return harmonize, deharmonize


harm_harmonize, harm_deharmonize = harmWrapper()
register(harm_harmonize)
register(harm_deharmonize)


# (de-)harmonize helper
def _outsortCrap(
    data, field, flagger, drop_flags=None
):

    """Harmonization gets the more easy, the more data points we can exclude from crowded sampling intervals.
    Depending on passed key word options the function will remove nan entries and as-suspicious-flagged values from
    the data and the flags passed. In deharmonization the function is used to reconstruct original flags field shape.


    :param data:            pd.Series. ['data'].
    :param flagger:         saqc.flagger.
    :param drop_list:       List or None. Default = None. List of flags that shall be dropped from data. If None is
                            passed (default), list based data dropping is omitted.
    :param return_drops:    Boolean. Default = False. If True, return the drops only. If False, return the data and
                            flags without drops.

    """
    assert isinstance(data, pd.Series), "data must be pd.Series"

    drop_mask = pd.Series(data=False, index=data.index)

    drop_flags = toSequence(drop_flags, default=flagger.BAD)
    for drop_flag in drop_flags:
        drop_mask = drop_mask | flagger.isFlagged(field, flag=drop_flag, comparator="==")

    flagger_out = flagger.slice(loc=~drop_mask)
    return data[~drop_mask], flagger_out, drop_mask


def _makeGrid(t0, t1, freq, name=None):
    """
    Returns a frequency grid, covering the date range of 'data'.
    :param data:    pd.Series. ['data']
    :param freq:    Offset String. Intended Sampling frequency.
    :return:        pd.Series. ['data'].
    """

    harm_start = t0.floor(freq=freq)
    harm_end = t1.ceil(freq=freq)
    return pd.date_range(start=harm_start, end=harm_end, freq=freq, name=name)


def _insertGrid(data, freq):
    """
    Depending on the frequency, the data has to be harmonized to, the passed data series gets reindexed with an index,
    containing the 'original' entries and additionally, if not present, the equidistant entries of the frequency grid.
    :param data:    pd.Series. ['data']
    :param freq:    Offset String. Intended Sampling frequency.
    :return:        pd.Series. ['data'].
    """

    return data.reindex(
        data.index.join(_makeGrid(data.index[0], data.index[-1], freq, name=data.index.name), how="outer",)
    )


def _interpolateGrid(
    data, freq, method, order=1, agg_method=sum, downcast_interpolation=False,
):
    """The function calculates grid point values for a passed pd.Series (['data']) by applying
    the selected interpolation/fill method. (passed to key word 'method'). The interpolation will apply for grid points
    only, that have preceding (forward-aggregation/forward-shifts) or succeeding (backward-aggregation/backward-shift)
    values, or both ("real" interpolations, like linear, polynomial, ...-> see documentation below).

    Data must be cleared from nans before entering here.

    Methods:
    All Methods calculate new values at grid points, if there is no data available for that sampling point.

    1. "real" INTERPOLATIONS

    There are available all the interpolation methods from the pandas.interpolate() method and they are applicable by
    the very same key words, that you would pass to pd.Series.interpolates's method parameter.

    Be careful with pd.Series.interpolate's 'nearest' and 'pad':
    To just fill grid points forward/backward or from the nearest point - and
    assign grid points, that refer to missing data, a nan value, the use of "fshift", "bshift" and "nshift" is
    recommended, to ensure the result expected. (The methods diverge in some special cases).

    To make an index-aware linear interpolation, "times" has to be passed - NOT 'linear'.

    2. SHIFTS:

    'fshift'        -  every grid point gets assigned its ultimately preceeding value - if there is one available in
                    the preceeding sampling interval.
    'bshift'        -  every grid point gets assigned its first succeeding value - if there is one available in the
                    succeeding sampling interval.
    'nshift' -  every grid point gets assigned the nearest value in its range ( range = +/-(freq/2) ).

    3. AGGREGATIONS

    'nagg'   - all values in the range (+/- freq/2) of a grid point get aggregated with agg_method and assigned
                    to it.
    'bagg'          - all values in a sampling interval get aggregated with agg_method and the result gets assigned to
                    the last grid point
    'fagg'          - all values in a sampling interval get aggregated with agg_method and the result gets assigned to
                    the next grid point

    :param data:        dios.DictOfSeries. ['data'].
    :param freq:        Offset String. the grid frequency.
    :param method:      String. Method you want to interpolate with. See function doc above.
    :param order:       Integer. Default = 1. If an interpolation method is selected that needs
                        to know about its "order", this is where you pass it. (For example 'polynomial', 'spline')
    :param agg_method:  Func. Default = sum. If an aggregation method is selected for grid point filling,
                        you need to pass the aggregation method to this very parameter. Note that it should be able
                        to handle empty argument series passed as well as np.nan passed.
    :return:            dios.DictOfSeries. ['data'].
    """

    chunk_bounds = None
    aggregations = ["nagg", "bagg", "fagg"]
    shifts = ["fshift", "bshift", "nshift"]
    interpolations = [
        "linear",
        "time",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "spline",
        "barycentric",
        "polynomial",
        "krogh",
        "piecewise_polynomial",
        "spline",
        "pchip",
        "akima",
    ]
    data = data.copy()

    # Aggregations:
    if method in aggregations:
        data = aggregate2Freq(data, method, freq, agg_method)

    # Shifts
    elif method in shifts:
        data = shift2Freq(data, method, freq)

    # Interpolations:
    elif method in interpolations:
        # account for annoying case of subsequent frequency alligned values, differing exactly by the margin
        # 2*freq:
        spec_case_mask = data.index.to_series()
        spec_case_mask = spec_case_mask - spec_case_mask.shift(1)
        spec_case_mask = spec_case_mask == 2 * pd.Timedelta(freq)
        spec_case_mask = spec_case_mask[spec_case_mask]
        spec_case_mask = spec_case_mask.resample(freq).asfreq().dropna()

        if not spec_case_mask.empty:
            spec_case_mask = spec_case_mask.tshift(-1, freq)

        data = _insertGrid(data, freq)
        data, chunk_bounds = interpolateNANs(
            data, method, order=order, inter_limit=2, downgrade_interpolation=downcast_interpolation,
            return_chunk_bounds=True
        )

        # exclude falsely interpolated values:
        data[spec_case_mask.index] = np.nan
        data = data.asfreq(freq)

    else:
        methods = "\n".join([", ".join(shifts), ", ".join(aggregations), ", ".join(interpolations)])
        raise ValueError(f"Unknown interpolation method: '{method}', please select from:\n{methods}")


    return data, chunk_bounds


def _reshapeFlags(
    flagger,
    field,
    freq,
    method="fshift",
    agg_method=max,
    missing_flag=None,
    set_shift_comment=True,
    block_flags=None,
    **kwargs,
):
    """To continue processing flags after harmonization/interpolation, old pre-harm flags have to be distributed onto
    new grid.

    There are the following methods available for flags projection. Note, that not every combination of flags projection
    and interpolation method will lead to useful results. (For example, interpolating with 'fshift' and projecting with
    bfill' would be a bad approach obviously.):

    Passed aggregation methods shall return a valid flag for empty sampling intervals, or the value np.nan
    - since np.nan values will be replaced by "missing_flag" anyway.

    'fshift'/'bshift'   - forward/backward projection. Only the very
                        first/last flag will be projected onto the last/next grid point. Extra flag fields like comment,
                        just get shifted along with the flag. Only inserted flags for empty intervals will take the
                        **kwargs argument.
                        Set 'set_shift_comment' to True,  to apply kwargs** to all flags (currently really slow)
    'fagg'/'bagg'       - All flags, referring to a sampling intervals measurements get aggregated forward/backward
                        with the agg_method selected.

    'nshift'     - every grid point gets assigned the nearest flag in its range
                        ( range = grid_point +/-(freq/2) ).Extra flag fields like comment,
                        just get shifted along with the flag. Only inserted flags for empty intervals will take the
                        **kwargs argument.

    'nagg'         - every grid point gets assigned the aggregation (by agg_method), of all the flags in its range.

    :param flagger:     saqc.flagger. The flagger, the passed flags frame refer to.
    :param method:      String. Default = 'fshift'. A methods keyword. (see func doc above)
    :param agg_method:  Func. Default = max. method, multiple flags shall be aggregated with, if an aggregation method
                        is selected for flags projection.
    :param missing_flag:Integer. Default = -1. If there were no flags referring to the harmonized interval, this
                        parameter determines wich flag will be inserted into the reshaped flags frame by selecting
                        flagger.flags[missing_flag]. The parameter defaults to the worst flag that can be thought of, in
                        terms of the used flagger.
    :param set_shift_comment:   Boolean. Default = False. The shifting methods for flags projection are really fast,
                        however, the methods used, do not allow to 'reflag' and apply eventually passed **kwargs.
                        Setting set_shift_comment to True, **kwargs will be applied, but the whole process will slow
                        down significantly.
    :block_flags:       DatetimeIndex. A DatetimeIndex containing labels that will get the "nan-flag" assigned.
                        This option mainly is introduced to account for the backtracking inconsistencies at the end
                        and beginning of interpolation chunks.
    :return: flags:     pd.Series/dios.DictOfSeries. The reshaped pandas like Flags object, referring to the harmonized data.
    """

    missing_flag = missing_flag or flagger.BAD
    aggregations = [
        "nagg",
        "bagg",
        "fagg",
        "nagg_no_deharm",
        "bagg_no_deharm",
        "fagg_no_deharm",
    ]
    shifts = ["fshift", "bshift", "nshift"]

    flags = flagger.getFlags()
    fdata = flags[field]

    if method in shifts:
        # forward/backward projection of every intervals last/first flag - rest will be dropped
        fdata = shift2Freq(fdata, method, freq)

        flags[field] = fdata
        flagger_new = flagger.initFlags(flags=flags)
        flagger_new.setFlags(field, loc=fdata.isna(), flag=missing_flag, force=True, **kwargs)

        if set_shift_comment:
            flagger_new = flagger_new.setFlags(field, flag=fdata, force=True, **kwargs)

    elif method in aggregations:
        fdata = aggregate2Freq(fdata, method, freq, agg_method, fill_value=missing_flag)
        fdata = fdata.astype(flagger.dtype)

        # block flagging/backtracking of chunk_starts/chunk_ends
        if block_flags is not None:
            fdata[block_flags] = np.nan

        flags[field] = fdata
        flagger_new = flagger.initFlags(flags=flags)

    else:
        methods = ", ".join(shifts + ["\n"] + aggregations)
        raise ValueError(
            "Passed reshaping method keyword:'{}', is unknown. Please select from: \n '{}'.".format(method, methods)
        )

    # block flagging/backtracking of chunk_starts/chunk_ends
    if block_flags is not None:
        flags_to_force = pd.Series(np.nan, index=block_flags).astype(flagger_new.dtype)
        flagger_new = flagger_new.setFlags(field, loc=block_flags, flag=flags_to_force, force=True,)
    return flagger_new


def _backtrackFlags(flagger_harmony, flagger_original_clean, flagger_original, freq, track_method="invert_fshift", co_flagging=False):

    # in the case of "real" up/downsampling - evaluating the harm flags against the original flags makes no sence!
    if track_method in ["regain"]:
        return flagger_original_clean

    flags_harmony = flagger_harmony.getFlags()
    flags_original_clean = flagger_original_clean.getFlags()
    flags_original = flagger_original.getFlags()

    flags_header = flags_harmony.columns
    assert len(flags_header) == 1

    flags_original_clean = flags_original_clean.squeeze()
    flags_original = flags_original.squeeze()
    flags_harmony = flags_harmony.squeeze()
    assert isinstance(flags_harmony, pd.Series)
    assert isinstance(flags_original_clean, pd.Series)

    if track_method in ["invert_fshift", "invert_bshift", "invert_nearest"] and co_flagging is True:
        if track_method == "invert_fshift":
            method = "bfill"
            tolerance = pd.Timedelta(freq)
        elif track_method == "invert_bshift":
            method = "ffill"
            tolerance = pd.Timedelta(freq)
        # var set for "invert nearest"
        else:
            # NOTE: co_flagging bug path
            method = "nearest"
            tolerance = pd.Timedelta(freq) / 2

        flags_harmony = flags_harmony.reindex(flags_original.index, method=method, tolerance=tolerance)
        replacement_mask = flags_harmony > flags_original
        flags_original.loc[replacement_mask] = flags_harmony.loc[replacement_mask]

    if track_method in ["invert_fshift", "invert_bshift", "invert_nearest"] and co_flagging is False:
        if track_method == "invert_fshift":
            method = "backward"
            tolerance = pd.Timedelta(freq)
        elif track_method == "invert_bshift":
            method = "forward"
            tolerance = pd.Timedelta(freq)
        # var set for 'invert nearest'
        else:
            method = "nearest"
            tolerance = pd.Timedelta(freq) / 2

        flags_harmony = pd.merge_asof(
            flags_harmony.to_frame(),
            pd.DataFrame(flags_original_clean.index.values,
                         index=flags_original_clean.index,
                         columns=["pre_index"]),
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            direction=method,
        )

        flags_harmony.dropna(subset=["pre_index"], inplace=True)
        flags_harmony.set_index(["pre_index"], inplace=True)
        # get rid of Dataframe (not dios !) , that we needed for the merge_asof()-method
        flags_harmony = flags_harmony.squeeze()

        replacement_mask = flags_harmony > flags_original_clean.loc[flags_harmony.index]
        flags_original_clean.loc[replacement_mask[replacement_mask].index] = flags_harmony.loc[replacement_mask]

        drops_index = flags_original.index.difference(flags_original_clean.index)
        flags_original = flags_original_clean.reindex(flags_original_clean.index.join(drops_index, how='outer'))
        flags_original.loc[drops_index] = flags_original[drops_index]

    res = dios.DictOfSeries(flags_original, columns=flags_header)
    return flagger_original.initFlags(flags=res)


@register
def harm_shift2Grid(data, field, flagger, freq, method="nshift", drop_flags=None, **kwargs):
    return harm_harmonize(
        data, field, flagger, freq, inter_method=method, reshape_method=method, drop_flags=drop_flags, **kwargs,
    )


@register
def harm_aggregate2Grid(
    data, field, flagger, freq, value_func, flag_func=np.nanmax, method="nagg", drop_flags=None, **kwargs,
):
    return harm_harmonize(
        data,
        field,
        flagger,
        freq,
        inter_method=method,
        reshape_method=method,
        inter_agg=value_func,
        reshape_agg=flag_func,
        drop_flags=drop_flags,
        **kwargs,
    )


@register
def harm_linear2Grid(data, field, flagger, freq, method="nagg", func=np.nanmax, drop_flags=None, **kwargs):
    return harm_harmonize(
        data,
        field,
        flagger,
        freq,
        inter_method="time",
        reshape_method=method,
        reshape_agg=func,
        drop_flags=drop_flags,
        **kwargs,
    )


@register
def harm_interpolate2Grid(
    data, field, flagger, freq, method, order=1, flag_method="nagg", flag_func=np.nanmax, drop_flags=None, **kwargs,
):
    return harm_harmonize(
        data,
        field,
        flagger,
        freq,
        inter_method=method,
        inter_order=order,
        reshape_method=flag_method,
        reshape_agg=flag_func,
        drop_flags=drop_flags,
        **kwargs,
    )


