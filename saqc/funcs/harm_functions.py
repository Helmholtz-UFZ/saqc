#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging

from saqc.funcs.functions import flagMissing


# todo: frequencie estimation function
# todo: tests!
# todo: flag assignment to wholes is slightly unconsistent since it doesnt get the missing keyword.
# todo: restrict parameter freedom (harm_freq and deharm_freq have to equal -> auto derive deharm_freq)
# todo: check agg_method argument behavior as passed to resample (nan handling / empty series handling)
# todo: drop-lists
# todo: rename set_shift_comment
# todo: rong method keyword error raise
# todo: make - full grid!
# todo: Bug in set shift comment!
# todo: accelerated func applies

def harm_wrapper(harm=True, heap={}):
    # NOTE:
    # (1) - harmonization will ALWAYS flag flagger.BAD all the np.nan values and afterwards DROP ALL
    #       flagger.BAD flagged values from flags frame for further flagging!!!!!!!!!!!!!!!!!!!!!
    harm_2_deharm = {'fshift': 'invert_fshift',
                     'bshift': 'invert_bshift',
                     'nearest_shift': 'invert_nearest',
                     'fagg': 'invert_fshift',
                     'bagg': 'invert_bshift',
                     'nearest_agg': 'invert_nearest'}

    def harmonize(data, flags, field, flagger, freq, inter_method, reshape_method, inter_agg=np.mean, inter_order=1,
                  inter_downcast=False,
                  reshape_agg=max, reshape_missing_flag_index=-1,
                  reshape_shift_comment=False, outsort_drop_susp=True, outsort_drop_list=None
                  , data_missing_value=np.nan, **kwargs):

        # for some tingle tangle reasons, resolving the harmonization will not be sound, if not all missing/np.nan
        # values get flagged initially:
        data, flags = flagMissing(data, flags, field, flagger, nodata=data_missing_value, **kwargs)

        # before sending the current flags and data frame to the future (for backtracking reasons), we clear it
        # from merge-nans that just resulted from harmonization of other variables!
        dat_col, flags_col = _fromMerged(data, flags, flagger, field)

        # now we send the flags frame in its current shape to the future:
        heap.update({field: {'original_data': flags_col.assign(data_values=dat_col)}})

        # to this subdict we add some configuration information for deharmonization
        heap[field].update({'freq': freq,
                            'reshape_method': reshape_method,
                            'drop_susp': outsort_drop_susp,
                            'drop_list': outsort_drop_list})

        # furthermore we need to memorize the initial timestamp to ensure output format will equal input format.
        if 'initial_ts' not in heap.keys():
            heap.update({'initial_ts': dat_col.index})

        # now we can manipulate it without loosing information gathered before harmonization
        dat_col, flags_col = _outsort_crap(dat_col, flags_col, field, flagger, drop_suspicious=outsort_drop_susp,
                                           drop_bad=True, drop_list=outsort_drop_list)
        # interpolation! (yeah)
        dat_col = _interpolate_grid(dat_col, freq, method=inter_method, order=inter_order, agg_method=inter_agg,
                                    total_range=(heap['initial_ts'][0], heap['initial_ts'][-1]),
                                    downcast_interpolation=inter_downcast)

        # flags now have to be carefully adjusted according to the changes/shifts we did to data
        flags_col = _reshape_flags(flags_col, flagger, field, ref_index=dat_col.index, method=reshape_method,
                                   agg_method=reshape_agg, missing_flag=reshape_missing_flag_index,
                                   set_shift_comment=reshape_shift_comment, **kwargs)

        # finally we happily blow up the data and flags frame again, to release them on their ongoing journey through
        # saqc.
        data, flags = _toMerged(data, flags, flagger, field, data_to_insert=dat_col, flags_to_insert=flags_col)

        return data, flags

    def deharmonize(data, flags, field, flagger, co_flagging=False, **kwargs):

        # Check if there is backtracking information available for actual harmonization resolving
        if field not in heap:
            logging.warning('No backtracking data for resolving harmonization of "{}". Reverse projection of flags gets'
                            ' skipped!'.format(field))
            return data, flags

        # get some deharm configuration infos from the heap:
        freq = heap[field]['freq']
        redrop_susp = heap[field]['drop_susp']
        redrop_list = heap[field]['drop_list']
        resolve_method = harm_2_deharm[heap[field]['reshape_method']]

        # retrieve data and flags from the merged saqc-conform data frame (and by that get rid of blow-up entries).
        dat_col, flags_col = _fromMerged(data, flags, flagger, field)

        # reconstruct the drops that were performed before harmonization
        drops, pre_flags = _outsort_crap(dat_col, heap[field]['original_data'].drop('data_values', axis=1, inplace=False), field, flagger,
                                         drop_suspicious=redrop_susp, drop_bad=True, drop_list=redrop_list,
                                         return_drops=True)

        # with reconstructed pre-harmonization flags-frame -> perform the projection of the flags calculated for
        # the harmonized timeseries, onto the original timestamps
        flags_col = _backtrack_flags(flags_col, pre_flags, flagger, freq, track_method=resolve_method,
                                     co_flagging=co_flagging)

        # now: re-insert the pre-harmonization-drops
        flags_col = flags_col.reindex(flags_col.index.join(drops.index, how='outer'))
        # due to assignment reluctants with 1-d-dataframes we are squeezing:
        flags_col = flags_col.squeeze(axis=1)
        drops = drops.squeeze(axis=1)
        flags_col.loc[drops.index] = drops
        # but to stick with the policy of always having flags as pd.DataFrames we blow up the flags col again:
        if isinstance(flags_col, pd.Series):
            flags_col = flags_col.to_frame()

        dat_col = heap[field]['original_data']['data_values'].reindex(flags_col.index, fill_value=np.nan)
        dat_col.name = field
        # transform the result into the form, data travels through saqc:
        data, flags = _toMerged(data, flags, flagger, field, dat_col, flags_col, target_index=heap['initial_ts'])
        # remove weight from the heap:
        heap.pop(field)
        # clear heap if nessecary:
        if (len(heap.keys()) == 1) and (list(heap.keys())[0] == 'initial_ts'):
            heap.pop('initial_ts')
        # bye bye data
        return data, flags

    if harm:
        return harmonize
    else:
        return deharmonize


def _outsort_crap(data, flags, field, flagger, drop_suspicious=True, drop_bad=True, drop_list=None, return_drops=False,
                  **kwargs):

    """Harmonization gets the more easy, the more data points we can exclude from crowded sampling intervals.
    Depending on passed key word options the function will remove nan entries and as-suspicious-flagged values from
    the data and the flags passed. In deharmonization the function is used to reconstruct original flags field shape.

    :param data:            pd.Series. ['data'].
    :param flags:           pd.PandasLike. The flags associated with the data passed.
    :param flagger:         saqc.flagger.
    :param drop_suspicious: Boolean. Default = True. If True, only values that are flagged GOOD or UNFLAGGED get
                            processed.
    :param drop_bad:        Boolean. Default = True. If True, BAD-flagged values get dropped from data.
    :param drop_list:       List or None. Default = None. List of flags that shall be dropped from data. If None is
                            passed (default), list based data dropping is omitted.
    :param return_drops:    Boolean. Default = False. If True, return the drops only. If False, return the data and
                            flags without drops.
    :return:                If return_drops=False. (default) Returns data, flags tuple with values-to-be-dropped
                            dropped.
                            If return_drops=True. Returns the dropped flags.
    """

    drop_mask = pd.Series(data=False, index=flags.index)

    if drop_bad is True:
        drop_mask = drop_mask | flagger.isFlagged(flags, field, flag=flagger.BAD, comparator='==')

    if drop_suspicious is True:
        drop_mask = drop_mask | ~(flagger.isFlagged(flags, field, flag=flagger.GOOD, comparator='<='))

    if drop_list is not None:
        for to_drop in drop_list:
            if to_drop in flagger.dtype.categories:
                drop_mask = drop_mask | flagger.isFlagged(flags, field, flag=to_drop, comparator='==')
            else:
                logging.warning('Cant drop "{}" - flagged data. Its not a flag value, the passed flagger happens to '
                                'know about.'.format(str(to_drop)))

    if return_drops:
        # return flag drops at first argument
        return flags[drop_mask], flags[~drop_mask]
    else:
        # return data at first argument
        return data[~drop_mask], flags[~drop_mask]


def _make_grid(t0, t1, freq):
    """
    Returns a frequency grid, covering the date range of 'data'.
    :param data:    pd.Series. ['data']
    :param freq:    Offset String. Intended Sampling frequency.
    :return:        pd.Series. ['data'].
    """


    harm_start = t0.floor(freq=freq)
    harm_end = t1.ceil(freq=freq)
    return pd.date_range(start=harm_start, end=harm_end, freq=freq)


def _insert_grid(data, freq):
    """
    Depending on the frequency, the data has to be harmonized to, the passed data series gets reindexed with an index,
    containing the 'original' entries and additionally, if not present, the equidistant entries of the frequency grid.
    :param data:    pd.Series. ['data']
    :param freq:    Offset String. Intended Sampling frequency.
    :return:        pd.Series. ['data'].
    """

    return data.reindex(data.index.join(_make_grid(data.index[0], data.index[-1], freq), how='outer'))


def _interpolate_grid(data, freq, method, order=1, agg_method=sum, total_range=None, downcast_interpolation=False):
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
    assign grid points, that refer to missing data, a nan value, the use of "fshift", "bshift" and "nearest_shift" is
    recommended, to ensure the result expected. (The methods diverge in some special cases).

    To make an index-aware linear interpolation, "times" has to be passed - NOT 'linear'.

    2. SHIFTS:

    'fshift'        -  every grid point gets assigned its ultimately preceeding value - if there is one available in
                    the preceeding sampling interval.
    'bshift'        -  every grid point gets assigned its first succeeding value - if there is one available in the
                    succeeding sampling interval.
    'nearest_shift' -  every grid point gets assigned the nearest value in its range ( range = +/-(freq/2) ).

    3. AGGREGATIONS

    'nearest_agg'   - all values in the range (+/- freq/2) of a grid point get aggregated with agg_method and assigned
                    to it.
    'bagg'          - all values in a sampling interval get aggregated with agg_method and the result gets assigned to
                    the last grid point
    'fagg'          - all values in a sampling interval get aggregated with agg_method and the result gets assigned to
                    the next grid point

    :param data:        pd.DataFrame. ['data'].
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
    :return:            pd.DataFrame. ['data'].
    """

    aggregations = ['nearest_agg', 'bagg', 'fagg']
    shifts = ['fshift', 'bshift', 'nearest_shift']
    interpolations = ['linear', 'time', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric',
            'polynomial', 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima']
    data = data.copy()
    ref_index = _make_grid(data.index[0], data.index[-1], freq)
    if total_range is not None:
        total_index = _make_grid(total_range[0], total_range[1], freq)

    # Aggregations:
    if method in aggregations:
        if method == 'nearest_agg':
            # all values within a grid points range (+/- freq/2, closed to the left) get aggregated with 'agg method'
            # some timestamp acrobatics to feed the base keyword properly
            seconds_total = pd.Timedelta(freq).total_seconds()
            seconds_string = str(int(seconds_total)) + 's'
            # calculate the series of aggregated values
            data = data.resample(seconds_string, base=seconds_total / 2,
                                 loffset=pd.Timedelta(freq) / 2).apply(agg_method)

        elif method == 'bagg':
            # all values in a sampling interval get aggregated with agg_method and assigned to the last grid point
            data = data.resample(freq).apply(agg_method)
        # if method is fagg
        else:
            # all values in a sampling interval get aggregated with agg_method and assigned to the next grid point
            data = data.resample(freq, closed='right', label='right').apply(agg_method)
        # some consistency cleanup:
        if total_range is None:
            data = data.reindex(ref_index)

    # Shifts
    elif method in shifts:
        if method == 'fshift':
            direction = 'ffill'
            tolerance = pd.Timedelta(freq)

        elif method == 'bshift':
            direction = 'bfill'
            tolerance = pd.Timedelta(freq)
        # if method = nearest_shift
        else:
            direction = 'nearest'
            tolerance = pd.Timedelta(freq)/2


        data = data.reindex(ref_index, method=direction, tolerance=tolerance)

    # Interpolations:
    elif method in interpolations:

        data = _insert_grid(data, freq)
        data = _interpolate(data, method, order=order, inter_limit=2, downcast_interpolation=downcast_interpolation)
        if total_range is None:
            data = data.asfreq(freq, fill_value=np.nan)

    else:
        methods = ", ".join(shifts + ['\n'] + aggregations + ['\n'] + interpolations)
        raise ValueError(
            "Passed interpolation method keyword:'{}', is unknown. Please select from: \n '{}'.".format(method,
                                                                                                        methods))
    if total_range is not None:
        data = data.reindex(total_index)

    return data


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
        gap_mask = gap_mask.replace(True, np.nan).fillna(method='bfill', limit=inter_limit)\
            .replace(np.nan, True).astype(bool)

    data = data[gap_mask]

    if method in ['linear', 'time']:

        data.interpolate(method=method, inplace=True, limit=1, limit_area='inside')

    else:

        gap_mask = (~gap_mask).cumsum()
        data = pd.merge(gap_mask, data, how='inner', left_index=True, right_index=True)

        def interpol_wrapper(x, wrap_order=order, wrap_method=method):
            if x.count() > wrap_order:
                try:
                    return x.interpolate(method=wrap_method, order=int(wrap_order))
                except (NotImplementedError, ValueError):
                    logging.warning('Interpolation with method {} is not supported at order {}. '
                                    'Interpolation will be performed with order {}'.
                                    format(method, str(wrap_order), str(wrap_order-1)))
                    return interpol_wrapper(x, int(wrap_order-1), wrap_method)
            elif x.size < 3:
                return x
            else:
                if downcast_interpolation:
                    return interpol_wrapper(x, int(x.count()-1), wrap_method)
                else:
                    return x

        data = data.groupby(data.columns[0]).transform(interpol_wrapper)

    return data


def _reshape_flags(flags, flagger, field, ref_index, method='fshift', agg_method=max, missing_flag=-1,
                   set_shift_comment=True, **kwargs):
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

    'nearest_shift'     - every grid point gets assigned the nearest flag in its range
                        ( range = grid_point +/-(freq/2) ).Extra flag fields like comment,
                        just get shifted along with the flag. Only inserted flags for empty intervals will take the
                        **kwargs argument.

    'nearest_agg'         - every grid point gets assigned the aggregation (by agg_method), of all the flags in its range.

    :param flags:       pd.Series/pd.DataFrame. The pandas like Flags object, referring to the data.
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
    :return: flags:     pd.Series/pd.DataFrame. The reshaped pandas like Flags object, referring to the harmonized data.
    """

    aggregations = ['nearest_agg', 'bagg', 'fagg']
    shifts = ['fshift', 'bshift', 'nearest_shift']

    freq = ref_index.freqstr
    # flags = flags[ref_index[0]:ref_index[-1]]

    if method in shifts:
        # forward/backward projection of every intervals last/first flag - rest will be dropped
        if method == 'fshift':
            direction = 'ffill'
            tolerance = pd.Timedelta(freq)

        elif method == 'bshift':
            direction = 'bfill'
            tolerance = pd.Timedelta(freq)
        # varset for nearest_shift
        else:
            direction = 'nearest'
            tolerance = pd.Timedelta(freq)/2

        flags = flags.reindex(ref_index, tolerance=tolerance, method=direction)

        # if you want to keep previous comments - only newly generated missing flags get commented:
        flags_series = flagger.getFlags(flags).squeeze()

        flags = flagger.setFlags(flags, field, loc=flags_series.isna(),
                             flag=flagger.dtype.categories[missing_flag], force=True, **kwargs)
        if set_shift_comment:
            flags = flagger.setFlags(flags, field,
                                     flag=flags_series, force=True, **kwargs)

    elif method in aggregations:
        # prepare resampling keywords
        if method == 'fagg':
            closed = 'right'
            label = 'right'
            base = 0
            freq_string = freq
        elif method == 'bagg':
            closed = 'left'
            label = 'left'
            base = 0
            freq_string = freq
        # var sets for 'nearest_agg':
        else:
            closed = 'left'
            label = 'left'
            seconds_total = pd.Timedelta(freq).total_seconds()
            base = seconds_total / 2
            freq_string = str(int(seconds_total)) + 's'
            if abs(flags.index[0] - flags.index[0].ceil(freq)) <= pd.Timedelta(freq) / 2:
                shift_correcture = -1
            else:
                shift_correcture = +1

        # resampling the flags series with aggregation method (squeezing because otherwise datetime index will get lost)
        flags = flagger.getFlags(flags).squeeze().resample(freq_string, closed=closed, label=label, base=base).apply(
            lambda x: agg_method(x) if not x.empty else flagger.dtype.categories[missing_flag]).astype(flagger.dtype)

        # restoring flags shape and column index:
        flags = flagger.setFlags(flagger.initFlags(flags.to_frame(name=field)), field,
                                 flag=flags, **kwargs)

        if method == 'nearest_agg':
            flags = flags.shift(periods=-shift_correcture, freq=pd.Timedelta(freq) / 2)

    else:
        methods = ", ".join(shifts + ['\n'] + aggregations)
        raise ValueError(
            "Passed reshaping method keyword:'{}', is unknown. Please select from: \n '{}'.".format(method,
                                                                                                    methods))

    return flags


def _backtrack_flags(flags_post, flags_pre, flagger, freq, track_method='invert_fshift', co_flagging=False):

    # circumvent "set value on copy warning":
    flags_pre = flags_pre.copy()
    flags_header = flags_post.columns

    if track_method in ['invert_fshift', 'invert_bshift', 'invert_nearest'] and co_flagging is True:
        if track_method == 'invert_fshift':
            method = 'bfill'
            tolerance = pd.Timedelta(freq)
        elif track_method == 'invert_bshift':
            method = 'ffill'
            tolerance = pd.Timedelta(freq)
        # var set for "invert nearest"
        else:
            method = 'nearest'
            tolerance = pd.Timedelta(freq)/2

        flags_post = flags_post.reindex(flags_pre.index, method=method, tolerance=tolerance)
        replacement_mask = flagger.getFlags(flags_post).squeeze() > flagger.getFlags(flags_pre).squeeze()
        # there is a mysterious problem when assigning 1-d-dataframes - so we squeeze:
        flags_pre = flags_pre.squeeze(axis=1)
        flags_post = flags_post.squeeze(axis=1)
        flags_pre.loc[replacement_mask] = flags_post.loc[replacement_mask]

    if track_method in ['invert_fshift', 'invert_bshift', 'invert_nearest'] and co_flagging is False:
        if track_method == 'invert_fshift':
            method = 'backward'
            tolerance = pd.Timedelta(freq)
        elif track_method == 'invert_bshift':
            method = 'forward'
            tolerance = pd.Timedelta(freq)
        # var set for 'invert nearest'
        else:
            method = 'nearest'
            tolerance = pd.Timedelta(freq)/2

        flags_post = pd.merge_asof(flags_post, pd.DataFrame(flags_pre.index.values, index=flags_pre.index,
                                                                columns=['pre_index']), left_index=True,
                                       right_index=True, tolerance=tolerance, direction=method)
        flags_post.dropna(subset=['pre_index'], inplace=True)
        flags_post.set_index(['pre_index'], inplace=True)

        # restore flag shape
        flags_post.columns = flags_header

        replacement_mask = flagger.getFlags(flags_post).squeeze() > \
                           flagger.getFlags(flags_pre.loc[flags_post.index, :]).squeeze()
        # there is a mysterious problem when assigning 1-d-dataframes - so we squeeze:
        flags_pre = flags_pre.squeeze(axis=1)
        flags_post = flags_post.squeeze(axis=1)
        flags_pre.loc[replacement_mask[replacement_mask].index] = flags_post.loc[replacement_mask]

    # sticking to the nomenklatura of always-DF for flags:
    if isinstance(flags_pre, pd.Series):
        flags_pre = flags_pre.to_frame()

    return flags_pre

def _fromMerged(data, flags, flagger, fieldname):
    # we need a not-na mask for the flags data to be retrieved:
    mask = flagger.getFlags(flags, fieldname).notna()

    # multi index case distinction:
    if isinstance(flags.columns, pd.MultiIndex):
        flags = flags.loc[:, flags.columns.get_level_values(0).isin([fieldname])]
    else:
        flags = flags[fieldname].to_frame(name=fieldname)

    data = data[fieldname]

    return data[mask], flags[mask]


def _toMerged(data, flags, flagger, fieldname, data_to_insert, flags_to_insert, target_index=None):

    data = data.copy()
    flags = flags.copy()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data.drop(fieldname, axis='columns', errors='ignore', inplace=True)
    flags.drop(fieldname, axis='columns', errors='ignore', inplace=True)

    # first case: there is no data, the data-to-insert would have to be merged with, and also are we not deharmonizing:
    if (flags.empty) & (target_index is None):
        return data_to_insert.to_frame(name=fieldname), flags_to_insert

    # if thats not the case: generate the drop mask for the remaining data:
    mask = (data.isna().all(axis=1))
    # we only want to drop lines, that do not have to be re-inserted in the merge:
    drops = mask[mask].index.difference(data_to_insert.index)
    # clear mask, but keep index
    mask[:] = True
    # final mask:
    mask[drops] = False

    # if we are not "de-harmonizing":
    if target_index is None:
        # erase nan rows in the data, that became redundant because of harmonization and merge with data-to-insert:
        data = pd.merge(data[mask], data_to_insert, how='outer', left_index=True, right_index=True)
        flags = pd.merge(flags[mask], flags_to_insert, how='outer', left_index=True, right_index=True)

    # if we are "de-harmonizing":
    else:
        # trivial case: there is only one variable:
        if flags.empty:
            data = data_to_insert.reindex(target_index).to_frame(name=fieldname)
            flags = flags_to_insert.reindex(target_index, fill_value=flagger.dtype.categories[0])
        # annoying case: more than one variables:
        # erase nan rows resulting from harmonization but keep/regain those, that were initially present in the data:
        else:
            new_index = data[mask].index.join(target_index, how='outer')
            data = data.reindex(new_index)
            flags = flags.reindex(new_index, fill_value=flagger.dtype.categories[0])
            data = pd.merge(data, data_to_insert, how='outer', left_index=True, right_index=True)
            flags = pd.merge(flags, flags_to_insert, how='outer', left_index=True, right_index=True)
            flags.fillna(flagger.dtype.categories[0], inplace=True)

        # internally harmonization memorizes its own manipulation by inserting nan flags -
        # those we will now assign the flagger.bad flag by the "missingTest":
        data, flags = flagMissing(data, flags, fieldname, flagger, nodata=np.nan)

    return data, flags





