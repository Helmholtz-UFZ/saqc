#! /usr/bin/env python
# -*- coding: utf-8 -*-
import functools
from typing import Optional, Tuple, Sequence, Callable, Optional
from typing_extensions import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from dios import DictOfSeries

from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.funcs.resampling import shift
from saqc.funcs.changepoints import assignChangePointCluster
from saqc.funcs.tools import drop, copy
from saqc.lib.tools import detectDeviants
from saqc.lib.types import FreqString, ColumnName, CurveFitter, TimestampColumnName
from saqc.lib.ts_operators import expModelFunc


LinkageString = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]


@register(masking='all', module="drift")
def flagDriftFromNorm(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        fields: Sequence[ColumnName],
        segment_freq: FreqString,
        norm_spread: float,
        norm_frac: float=0.5,
        metric: Callable[[np.ndarray, np.ndarray], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
        linkage_method: LinkageString="single",
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function flags value courses that significantly deviate from a group of normal value courses.

    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".

    See the Notes section for a more detailed presentation of the algorithm

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining which variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[[numpy.array, numpy.array], float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    Notes
    -----
    following steps are performed for every data "segment" of length `segment_freq` in order to find the
    "abnormal" data:

    1. Calculate the distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `fields`. (with :math:`d`
       denoting the distance function
       passed to the parameter `metric`.
    2. Calculate a dendogram with a hierarchical linkage algorithm, specified by the parameter `linkage_method`.
    3. Flatten the dendogram at the level, the agglomeration costs exceed the value given by the parameter `norm_spread`
    4. check if there is a cluster containing more than `norm_frac` percentage of the variables in fields.

        1. if yes: flag all the variables that are not in that cluster (inside the segment)
        2. if no: flag nothing

    The main parameter giving control over the algorithms behavior is the `norm_spread` parameter, that determines
    the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in every
    linkage step.
    For singleton clusters, that costs just equal half the distance, the timeseries in the clusters, have to
    each other. So, no timeseries can be clustered together, that are more then
    2*`norm_spread` distanted from each other.
    When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is
    calculated according to the linkage method specified by `linkage_method`. By default, it is the minimum distance,
    the members of the clusters have to each other.
    Having that in mind, it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).

    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """

    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)
    segments = data_to_flag.groupby(pd.Grouper(freq=segment_freq))
    for segment in segments:
        if segment[1].shape[0] <= 1:
            continue
        drifters = detectDeviants(segment[1], metric, norm_spread, norm_frac, linkage_method, 'variables')

        for var in drifters:
            flagger = flagger.setFlags(fields[var], loc=segment[1].index, **kwargs)

    return data, flagger


@register(masking='all', module="drift")
def flagDriftFromReference(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        fields: Sequence[ColumnName],
        segment_freq: FreqString,
        thresh: float,
        metric: Callable[[np.ndarray, np.ndarray], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function flags value courses that deviate from a reference course by a margin exceeding a certain threshold.

    The deviation is measured by the distance function passed to parameter metric.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    thresh : float
        The threshod by wich normal variables can deviate from the reference variable at max.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    Notes
    -----
    it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).
    """

    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)
    fields = list(fields)
    if field not in fields:
        fields.append(field)
    var_num = len(fields)
    segments = data_to_flag.groupby(pd.Grouper(freq=segment_freq))

    for segment in segments:

        if segment[1].shape[0] <= 1:
            continue
        for i in range(var_num):
            dist = metric(segment[1].iloc[:, i].values, segment[1].loc[:, field].values)
            if dist > thresh:
                flagger = flagger.setFlags(fields[i], loc=segment[1].index, **kwargs)

    return data, flagger


@register(masking='all', module="drift")
def flagDriftFromScaledNorm(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        fields_scale1: Sequence[ColumnName],
        fields_scale2: Sequence[ColumnName],
        segment_freq: FreqString,
        norm_spread: float,
        norm_frac: float=0.5,
        metric: Callable[[np.ndarray, np.ndarray], float]=lambda x, y: pdist(np.array([x, y]), metric='cityblock') / len(x),
        linkage_method: LinkageString="single",
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:


    """
    The function linearly rescales one set of variables to another set of variables with a different scale and then
    flags value courses that significantly deviate from a group of normal value courses.

    The two sets of variables can be linearly scaled one to another and hence the scaling transformation is performed
    via linear regression: A linear regression is performed on each pair of variables giving a slope and an intercept.
    The transformation is then calculated a the median of all the calculated slopes and intercepts.

    Once the transformation is performed, the function flags those values, that deviate from a group of normal values.
    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.
    fields_scale1 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 1.
    fields_scale2 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 2.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """

    fields = list(fields_scale1) + list(fields_scale2)
    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)

    convert_slope = []
    convert_intercept = []

    for field1 in fields_scale1:
        for field2 in fields_scale2:
            slope, intercept, _, _, _ = stats.linregress(data_to_flag[field1], data_to_flag[field2])
            convert_slope.append(slope)
            convert_intercept.append(intercept)

    factor_slope = np.median(convert_slope)
    factor_intercept = np.median(convert_intercept)

    dat = DictOfSeries()
    for field1 in fields_scale1:
        dat[field1] = factor_intercept + factor_slope * data_to_flag[field1]
    for field2 in fields_scale2:
        dat[field2] = data_to_flag[field2]

    dat_to_flag = dat[fields].to_df()

    segments = dat_to_flag.groupby(pd.Grouper(freq=segment_freq))
    for segment in segments:
        if segment[1].shape[0] <= 1:
            continue
        drifters = detectDeviants(segment[1], metric, norm_spread, norm_frac, linkage_method, 'variables')
        for var in drifters:
            flagger = flagger.setFlags(fields[var], loc=segment[1].index, **kwargs)

    return data, flagger


@register(masking='all', module="drift")
def correctExponentialDrift(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        maint_data_field: ColumnName,
        cal_mean: int=5,
        flag_maint_period: bool=False,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function fits an exponential model to chunks of data[field].
    It is assumed, that between maintenance events, there is a drift effect shifting the meassurements in a way, that
    can be described by the model M:

    M(t, a, b, c) = a + b(exp(c*t))

    Where as the values y_0 and y_1, describing the mean value directly after the last maintenance event (y_0) and
    directly before the next maintenance event (y_1), impose the following additional conditions on the drift model:.

    M(0, a, b, c) = y0
    M(1, a, b, c) = y1

    Solving the equation, one obtains the one-parameter Model:

    M_drift(t, c) = y0 + [(y1 - y0)/(exp(c) - )] * (exp(c*t) - 1)

    For every datachunk in between maintenance events.

    After having found the optimal parameter c*, the correction is performed by bending the fitted curve M_drift(t, c*),
    in a way that it matches y2 at t=1 (,with y2 being the mean value observed directly after the end of the next
    maintenance event).
    This bended curve is given by:

    M_shift(t, c*) = M(t, y0, [(y1 - y0)/(exp(c*) - )], c*)

    And the new values at t are computed via:

    new_vals(t) = old_vals(t) + M_shift(t) - M_drift(t)

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    maint_data_field : str
        The fieldname of the datacolumn holding the maintenance information.
        The maint data is to expected to have following form:
        The series' timestamp itself represents the beginning of a
        maintenance event, wheras the values represent the endings of the maintenance intervals.
    cal_mean : int, default 5
        The number of values the mean is computed over, for obtaining the value level directly after and
        directly before maintenance event. This values are needed for shift calibration. (see above description)
    flag_maint_period : bool, default False
        Wheather or not to flag BAD the values directly obtained while maintenance.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    # 1: extract fit intervals:
    if data[maint_data_field].empty:
        return data, flagger
    data = data.copy()
    to_correct = data[field]
    maint_data = data[maint_data_field]
    drift_frame = pd.DataFrame({"drift_group": np.nan, to_correct.name: to_correct.values}, index=to_correct.index)

    # group the drift frame
    for k in range(0, maint_data.shape[0] - 1):
        # assign group numbers for the timespans in between one maintenance ending and the beginning of the next
        # maintenance time itself remains np.nan assigned
        drift_frame.loc[maint_data.values[k] : pd.Timestamp(maint_data.index[k + 1]), "drift_group"] = k
    drift_grouper = drift_frame.groupby("drift_group")
    # define target values for correction
    shift_targets = drift_grouper.aggregate(lambda x: x[:cal_mean].mean()).shift(-1)

    for k, group in drift_grouper:
        dataSeries = group[to_correct.name]
        dataFit, dataShiftTarget = _drift_fit(dataSeries, shift_targets.loc[k, :][0], cal_mean)
        dataFit = pd.Series(dataFit, index=group.index)
        dataShiftTarget = pd.Series(dataShiftTarget, index=group.index)
        dataShiftVektor = dataShiftTarget - dataFit
        shiftedData = dataSeries + dataShiftVektor
        to_correct[shiftedData.index] = shiftedData

    if flag_maint_period:
        to_flag = drift_frame["drift_group"]
        to_flag = to_flag.drop(to_flag[: maint_data.index[0]].index)
        to_flag = to_flag[to_flag.isna()]
        flagger = flagger.setFlags(field, loc=to_flag, **kwargs)

    data[field] = to_correct

    return data, flagger


@register(masking='all', module="drift")
def correctRegimeAnomaly(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        cluster_field: ColumnName,
        model: CurveFitter,
        regime_transmission: Optional[FreqString]=None,
        x_date: bool=False
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function fits the passed model to the different regimes in data[field] and tries to correct
    those values, that have assigned a negative label by data[cluster_field].

    Currently, the only correction mode supported is the "parameter propagation."

    This means, any regime :math:`z`, labeled negatively and being modeled by the parameters p, gets corrected via:

    :math:`z_{correct} = z + (m(p^*) - m(p))`,

    where :math:`p^*` denotes the parameter set belonging to the fit of the nearest not-negatively labeled cluster.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    cluster_field : str
        A string denoting the field in data, holding the cluster label for the data you want to correct.
    model : Callable
        The model function to be fitted to the regimes.
        It must be a function of the form :math:`f(x, *p)`, where :math:`x` is the ``numpy.array`` holding the
        independent variables and :math:`p` are the model parameters that are to be obtained by fitting.
        Depending on the `x_date` parameter, independent variable x will either be the timestamps
        of every regime transformed to seconds from epoch, or it will be just seconds, counting the regimes length.
    regime_transmission : {None, str}, default None:
        If an offset string is passed, a data chunk of length `regime_transimission` right at the
        start and right at the end is ignored when fitting the model. This is to account for the
        unreliability of data near the changepoints of regimes.
    x_date : bool, default False
        If True, use "seconds from epoch" as x input to the model func, instead of "seconds from regime start".


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.
    """

    cluster_ser = data[cluster_field]
    unique_successive = pd.unique(cluster_ser.values)
    data_ser = data[field]
    regimes = data_ser.groupby(cluster_ser)
    para_dict = {}
    x_dict = {}
    x_mask = {}
    if regime_transmission is not None:
        # get seconds
        regime_transmission = pd.Timedelta(regime_transmission).total_seconds()
    for label, regime in regimes:
        if x_date is False:
            # get seconds data:
            xdata = (regime.index - regime.index[0]).to_numpy(dtype=float)*10**(-9)
        else:
            # get seconds from epoch data
            xdata = regime.index.to_numpy(dtype=float)*10**(-9)
        ydata = regime.values
        valid_mask = ~np.isnan(ydata)
        if regime_transmission is not None:
            valid_mask &= (xdata > xdata[0] + regime_transmission)
            valid_mask &= (xdata < xdata[-1] - regime_transmission)
        try:
            p, *_ = curve_fit(model, xdata[valid_mask], ydata[valid_mask])
        except (RuntimeError, ValueError):
            p = np.array([np.nan])
        para_dict[label] = p
        x_dict[label] = xdata
        x_mask[label] = valid_mask

    first_normal = unique_successive > 0
    first_valid = np.array([~pd.isna(para_dict[unique_successive[i]]).any() for i in range(0, unique_successive.shape[0])])
    first_valid = np.where(first_normal & first_valid)[0][0]
    last_valid = 1

    for k in range(0, unique_successive.shape[0]):
        if unique_successive[k] < 0 & (not pd.isna(para_dict[unique_successive[k]]).any()):
            ydata = data_ser[regimes.groups[unique_successive[k]]].values
            xdata = x_dict[unique_successive[k]]
            ypara = para_dict[unique_successive[k]]
            if k > 0:
                target_para = para_dict[unique_successive[k-last_valid]]
            else:
                # first regime has no "last valid" to its left, so we use first valid to the right:
                target_para = para_dict[unique_successive[k + first_valid]]
            y_shifted = ydata + (model(xdata, *target_para) - model(xdata, *ypara))
            data_ser[regimes.groups[unique_successive[k]]] = y_shifted
            if k > 0:
                last_valid += 1
        elif pd.isna(para_dict[unique_successive[k]]).any() & (k > 0):
            last_valid += 1
        else:
            last_valid = 1

    data[field] = data_ser
    return data, flagger


@register(masking='all', module="drift")
def correctOffset(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        max_mean_jump: float,
        normal_spread: float,
        search_winsz: FreqString,
        min_periods: int,
        regime_transmission: Optional[FreqString]=None,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    max_mean_jump : float
        when searching for changepoints in mean - this is the threshold a mean difference in the
        sliding window search must exceed to trigger changepoint detection.
    normal_spread : float
        threshold denoting the maximum, regimes are allowed to abolutely differ in their means
        to form the "normal group" of values.
    search_winsz : str
        Size of the adjacent windows that are used to search for the mean changepoints.
    min_periods : int
        Minimum number of periods a search window has to contain, for the result of the changepoint
        detection to be considered valid.
    regime_transmission : {None, str}, default None:
        If an offset string is passed, a data chunk of length `regime_transimission` right from the
        start and right before the end of any regime is ignored when calculating a regimes mean for data correcture.
        This is to account for the unrelyability of data near the changepoints of regimes.


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger
        The flagger object, holding flags and additional Informations related to `data`.

    """

    data, flagger = copy(data, field, flagger, field + '_CPcluster')
    data, flagger = assignChangePointCluster(
        data, field + '_CPcluster', flagger,
        lambda x, y: np.abs(np.mean(x) - np.mean(y)),
        lambda x, y: max_mean_jump,
        bwd_window=search_winsz,
        min_periods_bwd=min_periods
    )
    data, flagger = assignRegimeAnomaly(data, field, flagger, field + '_CPcluster', normal_spread)
    data, flagger = correctRegimeAnomaly(
        data, field, flagger, field + '_CPcluster',
        lambda x, p1: np.array([p1] * x.shape[0]),
        regime_transmission=regime_transmission
    )
    data, flagger = drop(data, field + '_CPcluster', flagger)

    return data, flagger


def _drift_fit(x, shift_target, cal_mean):
    x_index = x.index - x.index[0]
    x_data = x_index.total_seconds().values
    x_data = x_data / x_data[-1]
    y_data = x.values
    origin_mean = np.mean(y_data[:cal_mean])
    target_mean = np.mean(y_data[-cal_mean:])

    def modelWrapper(x, c, a=origin_mean, target_mean=target_mean):
        # final fitted curves val = target mean
        b = (target_mean - a) / (np.exp(c) - 1)
        return expModelFunc(x, a, b, c)

    dataFitFunc = functools.partial(modelWrapper, a=origin_mean, target_mean=target_mean)

    try:
        fitParas, *_ = curve_fit(dataFitFunc, x_data, y_data, bounds=([0], [np.inf]))
        dataFit = dataFitFunc(x_data, fitParas[0])
        b_val = (shift_target - origin_mean) / (np.exp(fitParas[0]) - 1)
        dataShiftFunc = functools.partial(expModelFunc, a=origin_mean, b=b_val, c=fitParas[0])
        dataShift = dataShiftFunc(x_data)
    except RuntimeError:
        dataFit = np.array([0] * len(x_data))
        dataShift = np.array([0] * len(x_data))

    return dataFit, dataShift


@register(masking='all', module="drift")
def flagRegimeAnomaly(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        cluster_field: ColumnName,
        norm_spread: float,
        linkage_method: LinkageString="single",
        metric: Callable[[np.ndarray, np.ndarray], float]=lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y)),
        norm_frac: float=0.5,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    A function to flag values belonging to an anomalous regime regarding modelling regimes of field.

    "Normality" is determined in terms of a maximum spreading distance, regimes must not exceed in respect
    to a certain metric and linkage method.

    In addition, only a range of regimes is considered "normal", if it models more then `norm_frac` percentage of
    the valid samples in "field".

    Note, that you must detect the regime changepoints prior to calling this function.

    Note, that it is possible to perform hypothesis tests for regime equality by passing the metric
    a function for p-value calculation and selecting linkage method "complete".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    cluster_field : str
        The name of the column in data, holding the cluster labels for the samples in field. (has to be indexed
        equal to field)
    norm_spread : float
        A threshold denoting the valuelevel, up to wich clusters a agglomerated.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the variables.
    metric : Callable[[numpy.array, numpy.array], float], default lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes. Defaults to just the difference in mean.
    norm_frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.

    Returns
    -------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    """

    data, flagger = assignRegimeAnomaly(data, field, flagger, cluster_field, norm_spread,
                                        linkage_method=linkage_method, metric=metric, norm_frac=norm_frac,
                                        set_cluster=False, set_flags=True, **kwargs)

    return data, flagger


@register(masking='all', module="drift")
def assignRegimeAnomaly(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        cluster_field: ColumnName,
        norm_spread: float,
        linkage_method: LinkageString="single",
        metric: Callable[[np.array, np.array], float]=lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y)),
        norm_frac: float=0.5,
        set_cluster: bool=True,
        set_flags: bool=False,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    A function to detect values belonging to an anomalous regime regarding modelling regimes of field.

    The function changes the value of the regime cluster labels to be negative.

    "Normality" is determined in terms of a maximum spreading distance, regimes must not exceed in respect
    to a certain metric and linkage method.

    In addition, only a range of regimes is considered "normal", if it models more then `norm_frac` percentage of
    the valid samples in "field".

    Note, that you must detect the regime changepoints prior to calling this function. (They are expected to be stored
    parameter `cluster_field`.)

    Note, that it is possible to perform hypothesis tests for regime equality by passing the metric
    a function for p-value calculation and selecting linkage method "complete".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    cluster_field : str
        The name of the column in data, holding the cluster labels for the samples in field. (has to be indexed
        equal to field)
    norm_spread : float
        A threshold denoting the valuelevel, up to wich clusters a agglomerated.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the variables.
    metric : Callable[[numpy.array, numpy.array], float], default lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes. Defaults to just the difference in mean.
    norm_frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.
    set_cluster : bool, default False
        If True, all data, considered "anormal", gets assigned a negative clusterlabel. This option
        is present for further use (correction) of the anomaly information.
    set_flags : bool, default True
        Wheather or not to flag abnormal values (do not flag them, if you want to correct them
        afterwards, becasue flagged values usually are not visible in further tests.).

    Returns
    -------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    """

    clusterser = data[cluster_field]
    cluster = np.unique(clusterser)
    cluster_dios = DictOfSeries({i: data[field][clusterser == i] for i in cluster})
    plateaus = detectDeviants(cluster_dios, metric, norm_spread, norm_frac, linkage_method, 'samples')

    if set_flags:
        for p in plateaus:
            flagger = flagger.setFlags(field, loc=cluster_dios.iloc[:, p].index, **kwargs)

    if set_cluster:
        for p in plateaus:
            if cluster[p] > 0:
                clusterser[clusterser == cluster[p]] = -cluster[p]

    data[cluster_field] = clusterser
    return data, flagger
