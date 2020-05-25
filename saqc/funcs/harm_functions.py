#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import dios

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
            total_range=(dat_col.index[0], dat_col.index[-1]),
            downcast_interpolation=inter_downcast,
        )

        # flags now have to be carefully adjusted according to the changes/shifts we did to data
        flagger_merged_clean_reshaped = _reshapeFlags(
            flagger_merged_clean,
            field,
            ref_index=dat_col.index,
            method=reshape_method,
            agg_method=reshape_agg,
            missing_flag=reshape_missing_flag,
            set_shift_comment=reshape_shift_comment,
            block_flags=chunk_bounds,
            **kwargs,
        )

        # TODO: ask BERT: why so verbose?
        # TODO: ask BERT: why so assert?
        flags_col = flagger_merged_clean_reshaped.getFlags(field)
        flags = flagger.getFlags()
        flags[field] = flags_col
        data[field] = dat_col
        flagger_out = flagger.initFlags(flags=flags)
        assert (data[field].index == flagger_out.getFlags(field).index).all()
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
        flags = flagger.getFlags()
        flags[field] = flags_col
        data[field] = dat_col
        flagger_out = flagger.initFlags(flags=flags)
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
    data, freq, method, order=1, agg_method=sum, total_range=None, downcast_interpolation=False,
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
    :param total_range  2-Tuple of pandas Timestamps.
                        The total range of all the data in the Dataframe that is currently processed. If not
                        None, the resulting harmonization grid of the current data column will range over the total
                        Data-range. This ensures not having nan-entries in the flags dataframe after harmonization.
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
    ref_index = _makeGrid(data.index[0], data.index[-1], freq, name=data.index.name)
    if total_range is not None:
        total_index = _makeGrid(total_range[0], total_range[1], freq, name=data.index.name)

    # Aggregations:
    if method in aggregations:
        if method == "nagg":
            # all values within a grid points range (+/- freq/2, closed to the left) get aggregated with 'agg method'
            # some timestamp acrobatics to feed the base keyword properly
            seconds_total = pd.Timedelta(freq).total_seconds()
            seconds_string = str(int(seconds_total)) + "s"
            # calculate the series of aggregated values
            data = data.resample(seconds_string, base=seconds_total / 2, loffset=pd.Timedelta(freq) / 2).apply(
                agg_method
            )

        elif method == "bagg":
            # all values in a sampling interval get aggregated with agg_method and assigned to the last grid point
            data = data.resample(freq).apply(agg_method)
        # if method is fagg
        else:
            # all values in a sampling interval get aggregated with agg_method and assigned to the next grid point
            data = data.resample(freq, closed="right", label="right").apply(agg_method)
        # some consistency cleanup:
        if total_range is None:
            data = data.reindex(ref_index)

    # Shifts
    elif method in shifts:
        if method == "fshift":
            direction = "ffill"
            tolerance = pd.Timedelta(freq)

        elif method == "bshift":
            direction = "bfill"
            tolerance = pd.Timedelta(freq)
        # if method = nshift
        else:
            direction = "nearest"
            tolerance = pd.Timedelta(freq) / 2

        data = data.reindex(ref_index, method=direction, tolerance=tolerance)

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
        data, chunk_bounds = _interpolate(
            data, method, order=order, inter_limit=2, downcast_interpolation=downcast_interpolation,
        )

        # exclude falsely interpolated values:
        data[spec_case_mask.index] = np.nan

        if total_range is None:
            data = data.asfreq(freq, fill_value=np.nan)

    else:
        methods = "\n".join([", ".join(shifts), ", ".join(aggregations), ", ".join(interpolations)])
        raise ValueError(f"Unknown interpolation method: '{method}', please select from:\n{methods}")

    if total_range is not None:
        data = data.reindex(total_index)

    return data, chunk_bounds


def _interpolate(data, method, order=2, inter_limit=2, downcast_interpolation=False):
    """
    The function interpolates nan-values (and nan-grids) in timeseries data. It can be passed all the method keywords
    from the pd.Series.interpolate method and will than apply this very methods. Note, that the inter_limit keyword
    really restricts the interpolation to chunks, not containing more than "inter_limit" nan entries
    (thereby opposing the limit keyword of pd.Series.interpolate).

    :param data:                    pd.Series. The data series to be interpolated
    :param method:                  String. Method keyword designating interpolation method to use.
    :param order:                   Integer. If your desired interpolation method needs an order to be passed -
                                    here you pass it.
    :param inter_limit:             Integer. Default = 2. Limit up to wich nan - gaps in the data get interpolated.
                                    Its default value suits an interpolation that only will apply on an inserted
                                    frequency grid.
    :param downcast_interpolation:  Boolean. Default False. If True:
                                    If a data chunk not contains enough values for interpolation of the order "order",
                                    the highest order possible will be selected for that chunks interpolation."
    :return:
    """

    gap_mask = (data.rolling(inter_limit, min_periods=0).apply(lambda x: np.sum(np.isnan(x)), raw=True)) != inter_limit

    if inter_limit == 2:
        gap_mask = gap_mask & gap_mask.shift(-1, fill_value=True)
    else:
        gap_mask = (
            gap_mask.replace(True, np.nan).fillna(method="bfill", limit=inter_limit).replace(np.nan, True).astype(bool)
        )
    # start end ending points of interpolation chunks have to be memorized to block their flagging:
    chunk_switches = gap_mask.astype(int).diff()
    chunk_starts = chunk_switches[chunk_switches == -1].index
    chunk_ends = chunk_switches[(chunk_switches.shift(-1) == 1)].index
    chunk_bounds = chunk_starts.join(chunk_ends, how="outer", sort=True)

    data = data[gap_mask]

    if method in ["linear", "time"]:

        data.interpolate(method=method, inplace=True, limit=1, limit_area="inside")

    else:
        dat_name = data.name
        gap_mask = (~gap_mask).cumsum()
        data = pd.merge(gap_mask, data, how="inner", left_index=True, right_index=True)

        def _interpolWrapper(x, wrap_order=order, wrap_method=method):
            if x.count() > wrap_order:
                try:
                    return x.interpolate(method=wrap_method, order=int(wrap_order))
                except (NotImplementedError, ValueError):
                    logger.warning(
                        "Interpolation with method {} is not supported at order {}. "
                        "Interpolation will be performed with order {}".format(
                            method, str(wrap_order), str(wrap_order - 1)
                        )
                    )
                    return _interpolWrapper(x, int(wrap_order - 1), wrap_method)
            elif x.size < 3:
                return x
            else:
                if downcast_interpolation:
                    return _interpolWrapper(x, int(x.count() - 1), wrap_method)
                else:
                    return x

        data = data.groupby(data.columns[0]).transform(_interpolWrapper)
        # squeezing the 1-dimensional frame resulting from groupby for consistency reasons
        data = data.squeeze(axis=1)
        data.name = dat_name
    return data, chunk_bounds


def _reshapeFlags(
    flagger,
    field,
    ref_index,
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

    freq = ref_index.freq

    # fixme: NOTE: now with dios we just work on the series in question and leave
    #  other indexes untouched...
    flags = flagger.getFlags()
    fdata = flags[field]

    if method in shifts:
        # forward/backward projection of every intervals last/first flag - rest will be dropped
        if method == "fshift":
            direction = "ffill"
            tolerance = pd.Timedelta(freq)

        elif method == "bshift":
            direction = "bfill"
            tolerance = pd.Timedelta(freq)
        # varset for nshift
        else:
            direction = "nearest"
            tolerance = pd.Timedelta(freq) / 2

        # if you want to keep previous comments
        # only newly generated missing flags get commented:

        fdata = fdata.reindex(ref_index, tolerance=tolerance, method=direction, fill_value=np.nan)

        flags[field] = fdata
        flagger_new = flagger.initFlags(flags=flags)
        flagger_new.setFlags(field, loc=fdata.isna(), flag=missing_flag, force=True, **kwargs)

        if set_shift_comment:
            flagger_new = flagger_new.setFlags(field, flag=fdata, force=True, **kwargs)

    elif method in aggregations:
        # prepare resampling keywords
        if method in ["fagg", "fagg_no_deharm"]:
            closed = "right"
            label = "right"
            base = 0
            freq_string = freq
        elif method in ["bagg", "bagg_no_deharm"]:
            closed = "left"
            label = "left"
            base = 0
            freq_string = freq
        # var sets for 'nagg':
        else:
            closed = "left"
            label = "left"
            seconds_total = pd.Timedelta(freq).total_seconds()
            base = seconds_total / 2
            freq_string = str(int(seconds_total)) + "s"
            i_start = fdata.index[0]
            if abs(i_start - i_start.floor(freq)) <= pd.Timedelta(freq) / 2:
                shift_correcture = 1
            else:
                shift_correcture = -1

        # resampling the flags series with aggregation method
        agg = lambda x: agg_method(x) if not x.empty else missing_flag
        resampled = fdata.resample(freq_string, closed=closed, label=label, base=base)
        # NOTE: breaks for non categorical flaggers
        fdata = resampled.apply(agg).astype(flagger.dtype)

        if method == "nagg":
            fdata = fdata.shift(periods=shift_correcture, freq=pd.Timedelta(freq) / 2)

        # some consistency clean up to ensure new flags frame matching new data frames size:
        if ref_index[0] != fdata.index[0]:
            fdata = pd.Series(data=flagger.BAD, index=[ref_index[0]]).astype(flagger.dtype).append(fdata)
        if ref_index[-1] != fdata.index[-1]:
            fdata = fdata.append(pd.Series(data=flagger.BAD, index=[ref_index[-1]]).astype(flagger.dtype))

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


@register
def harm_downsample(
    data,
    field,
    flagger,
    sample_freq,
    agg_freq,
    sample_func=np.nanmean,
    agg_func=np.nanmean,
    invalid_flags=None,
    max_invalid=None,
    **kwargs,
):


    if max_invalid is None:
        max_invalid = np.inf


    # define the "fastest possible" aggregator
    if sample_func is None:
        if max_invalid < np.inf:
            def aggregator(x):
                if x.isna().sum() < max_invalid:
                    return agg_func(x)
                else:
                    return np.nan
        else:
            def aggregator(x):
                return agg_func(x)

    else:
        dummy_resampler = pd.Series(np.nan, index=[pd.Timedelta("1min")]).resample("1min")
        if hasattr(dummy_resampler, sample_func.__name__):

            sample_func_name = sample_func.__name__
            if max_invalid < np.inf:

                def aggregator(x):
                    y = getattr(x.resample(sample_freq), sample_func_name)()
                    if y.isna().sum() < max_invalid:
                        return agg_func(y)
                    else:
                        return np.nan

            else:

                def aggregator(x):
                    return agg_func(getattr(x.resample(sample_freq), sample_func_name)())
        else:
            if max_invalid < np.inf:

                def aggregator(x):
                    y = x.resample(sample_freq).apply(sample_func)
                    if y.isna().sum() < max_invalid:
                        return agg_func(y)
                    else:
                        return np.nan
            else:
                def aggregator(x):
                    return agg_func(x.resample(sample_freq).apply(sample_func))

    return harm_harmonize(
        data,
        field,
        flagger,
        agg_freq,
        inter_method="bagg",
        reshape_method="bagg_no_deharm",
        inter_agg=aggregator,
        reshape_agg=np.nanmax,
        drop_flags=invalid_flags,
        **kwargs,
    )
