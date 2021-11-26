#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import functools
from typing import Optional, Tuple, Sequence, Callable, Optional
from typing_extensions import Literal

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from dios import DictOfSeries
from saqc.constants import BAD
from saqc.core.register import register, flagging
from saqc.core import Flags
from saqc.funcs.changepoints import _assignChangePointCluster
from saqc.funcs.tools import dropField, copyField

from saqc.lib.tools import detectDeviants, toSequence, filterKwargs
from saqc.lib.ts_operators import linearDriftModel, expDriftModel
from saqc.lib.types import CurveFitter
from saqc.lib.ts_operators import linearDriftModel, expDriftModel


LinkageString = Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
]

MODELDICT = {"linear": linearDriftModel, "exponential": expDriftModel}


@flagging()
def flagDriftFromNorm(
    data: DictOfSeries,
    field: Sequence[str],
    flags: Flags,
    freq: str,
    spread: float,
    frac: float = 0.5,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
        np.array([x, y]), metric="cityblock"
    )
    / len(x),
    method: LinkageString = "single",
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function flags value courses that significantly deviate from a group of normal value courses.

    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `frac` percent of the
    variables in "field".

    See the Notes section for a more detailed presentation of the algorithm

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : list of str
        List of fieldnames in data, determining which variables are to be included into the flagging process.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[[numpy.array, numpy.array], float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the input flags.

    Notes
    -----
    following steps are performed for every data "segment" of length `freq` in order to find the
    "abnormal" data:

    1. Calculate the distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `field`. (with :math:`d`
       denoting the distance function
       passed to the parameter `metric`.
    2. Calculate a dendogram with a hierarchical linkage algorithm, specified by the parameter `method`.
    3. Flatten the dendogram at the level, the agglomeration costs exceed the value given by the parameter `spread`
    4. check if there is a cluster containing more than `frac` percentage of the variables in field.

        1. if yes: flag all the variables that are not in that cluster (inside the segment)
        2. if no: flag nothing

    The main parameter giving control over the algorithms behavior is the `spread` parameter, that determines
    the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in every
    linkage step.
    For singleton clusters, that costs just equal half the distance, the timeseries in the clusters, have to
    each other. So, no timeseries can be clustered together, that are more then
    2*`spread` distanted from each other.
    When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is
    calculated according to the linkage method specified by `method`. By default, it is the minimum distance,
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
    fields = toSequence(field)
    data_to_flag = data[fields].to_df()
    data_to_flag.dropna(inplace=True)

    segments = data_to_flag.groupby(pd.Grouper(freq=freq))
    for segment in segments:

        if segment[1].shape[0] <= 1:
            continue

        drifters = detectDeviants(segment[1], metric, spread, frac, method, "variables")

        for var in drifters:
            flags[segment[1].index, fields[var]] = flag

    return data, flags


# default multivariate flagging
@register(
    mask=["field", "reference"],
    demask=["field", "reference"],
    squeeze=["field", "reference"],  # reference is written !
    multivariate=True,
    handles_target=True,
)
def flagDriftFromReference(
    data: DictOfSeries,
    field: Sequence[str],
    flags: Flags,
    reference: str,
    freq: str,
    thresh: float,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
        np.array([x, y]), metric="cityblock"
    )
    / len(x),
    target=None,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function flags value courses that deviate from a reference course by a margin exceeding a certain threshold.

    The deviation is measured by the distance function passed to parameter metric.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : list of str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    reference : str
        The reference variable, the deviation from wich determines the flagging.
    freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    thresh : float
        The threshod by wich normal variables can deviate from the reference variable at max.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    target : None
        ignored.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the input flags.

    Notes
    -----
    it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).
    """
    if target:
        raise NotImplementedError("target is not implemented for this function")

    fields = toSequence(field)
    data_to_flag = data[fields].to_df().dropna()

    fields = list(fields)
    if reference not in fields:
        fields.append(reference)

    segments = data_to_flag.groupby(pd.Grouper(freq=freq))
    for segment in segments:

        if segment[1].shape[0] <= 1:
            continue

        for i in range(len(fields)):
            dist = metric(
                segment[1].iloc[:, i].values, segment[1].loc[:, reference].values
            )

            if dist > thresh:
                flags[segment[1].index, fields[i]] = flag

    return data, flags


# default multivariate flagging
@register(
    mask=["set_1", "set_2"],
    demask=["set_1", "set_2"],
    squeeze=["set_1", "set_2"],
    multivariate=True,
    handles_target=True,
)
def flagDriftFromScaledNorm(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    set_1: Sequence[str],
    set_2: Sequence[str],
    freq: str,
    spread: float,
    frac: float = 0.5,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
        np.array([x, y]), metric="cityblock"
    )
    / len(x),
    method: LinkageString = "single",
    target: str = None,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function linearly rescales one set of variables to another set of variables
    with a different scale and then flags value courses that significantly deviate
    from a group of normal value courses.

    The two sets of variables can be linearly scaled one to another and hence the
    scaling transformation is performed via linear regression: A linear regression is
    performed on each pair of variables giving a slope and an intercept. The
    transformation is then calculated a the median of all the calculated slopes and
    intercepts.

    Once the transformation is performed, the function flags those values,
    that deviate from a group of normal values. "Normality" is determined in terms of
    a maximum spreading distance, that members of a normal group must not exceed. In
    addition, only a group is considered "normal" if it contains more then `frac`
    percent of the variables in "fields".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        A dummy parameter.

    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.

    set_1 : str
        The first set of fieldnames in data to be included into the flagging process.

    set_2 : str
        The second set of fieldnames in data to be included into the flagging process.

    freq : str
        An offset string, determining the size of the seperate datachunks that the
        algorihm is to be piecewise applied on.

    spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the
        "normal" group. See Notes section for more details.

    frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables,
        the "normal" group has to comprise to be the normal group actually. The
        higher that value, the more stable the algorithm will be with respect to
        false positives. Also, nobody knows what happens, if this value is below 0.5.

    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and
        return a float scalar value. This value is interpreted as the distance of the
        two input arrays. The default is the averaged manhatten metric. See the Notes
        section to get an idea of why this could be a good choice.

    method : str, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the
        timeseries. See the Notes section for more details. The keyword gets passed
        on to scipy.hierarchy.linkage. See its documentation to learn more about the
        different keywords (References [1]). See wikipedia for an introduction to
        hierarchical clustering (References [2]).

    target : None
        ignored

    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the input flags.

    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """
    if target:
        raise NotImplementedError("target is not implemented for this function")

    fields = list(set_1) + list(set_2)
    data_to_flag = data[fields].to_df().dropna()

    convert_slope = []
    convert_intercept = []

    for field1 in set_1:
        for field2 in set_2:
            slope, intercept, *_ = stats.linregress(
                data_to_flag[field1], data_to_flag[field2]
            )
            convert_slope.append(slope)
            convert_intercept.append(intercept)

    factor_slope = np.median(convert_slope)
    factor_intercept = np.median(convert_intercept)

    dat = DictOfSeries()
    for field1 in set_1:
        dat[field1] = factor_intercept + factor_slope * data_to_flag[field1]
    for field2 in set_2:
        dat[field2] = data_to_flag[field2]

    dat_to_flag = dat[fields].to_df()

    segments = dat_to_flag.groupby(pd.Grouper(freq=freq))
    for segment in segments:

        if segment[1].shape[0] <= 1:
            continue

        drifters = detectDeviants(segment[1], metric, spread, frac, method, "variables")

        for var in drifters:
            flags[segment[1].index, fields[var]] = flag

    return data, flags


@register(mask=["field"], demask=[], squeeze=[])
def correctDrift(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    maintenance_field: str,
    model: Callable[..., float] | Literal["linear", "exponential"],
    cal_range: int = 5,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function corrects drifting behavior.

    See the Notes section for an overview over the correction algorithm.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flags : saqc.Flags
        Container to store quality flags to data.
    maintenance_field : str
        The fieldname of the datacolumn holding the support-points information.
        The maint data is to expected to have following form:
        The series' timestamp itself represents the beginning of a
        maintenance event, wheras the values represent the endings of the maintenance intervals.
    model : Callable or {'exponential', 'linear'}
        A modelfunction describing the drift behavior, that is to be corrected.
        Either use built-in exponential or linear drift model by passing a string, or pass a custom callable.
        The model function must always contain the keyword parameters 'origin' and 'target'.
        The starting parameter must always be the parameter, by wich the data is passed to the model.
        After the data parameter, there can occure an arbitrary number of model calibration arguments in
        the signature.
        See the Notes section for an extensive description.
    cal_range : int, default 5
        The number of values the mean is computed over, for obtaining the value level directly after and
        directly before maintenance event. This values are needed for shift calibration. (see above description)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data

    Notes
    -----
    It is assumed, that between support points, there is a drift effect shifting the meassurements in a way, that
    can be described, by a model function ``M(t, *p, origin, target)``. (With ``0<=t<=1``, ``p`` being a parameter
    set, and ``origin``, ``target`` being floats).

    Note, that its possible for the model to have no free parameters p at all. (linear drift mainly)

    The drift model, directly after the last support point (``t=0``),
    should evaluate to the origin - calibration level (``origin``), and directly before the next support point
    (``t=1``), it should evaluate to the target calibration level (``target``).::

        M(0, *p, origin, target) = origin
        M(1, *p, origin, target) = target

    The model is than fitted to any data chunk in between support points, by optimizing the parameters ``p*``, and
    thus, obtaining optimal parameterset ``P*``.

    The new values at t are computed via:::

        new_vals(t) = old_vals(t) + M(t, *P, origin, target) - M_drift(t, *P, origin, new_target)

    Wheras ``new_target`` represents the value level immediately after the next support point.

    Examples
    --------
    Some examples of meaningful driftmodels.

    Linear drift modell (no free parameters).

    >>> M = lambda t, origin, target: origin + t * target

    exponential drift model (exponential raise!)

    >>> expFunc = lambda t, a, b, c: a + b * (np.exp(c * x) - 1)
    >>> M = lambda t, p, origin, target: expFunc(t, (target - origin) / (np.exp(abs(c)) - 1), abs(c))

    Exponential and linear driftmodels are part of the ``ts_operators`` library, under the names
    ``expDriftModel`` and ``linearDriftModel``.

    """
    # extract model func:
    if isinstance(model, str):
        if model not in MODELDICT:
            raise ValueError(
                f"invalid model '{model}', choose one of '{MODELDICT.keys()}'"
            )
        model = MODELDICT[model]

    # 1: extract fit intervals:
    if data[maintenance_field].empty:
        return data, flags

    to_correct = data[field].copy()
    maint_data = data[maintenance_field].copy()

    to_correct_clean = to_correct.dropna()
    d = {"drift_group": np.nan, to_correct.name: to_correct_clean.values}
    drift_frame = pd.DataFrame(d, index=to_correct_clean.index)

    # group the drift frame
    for k in range(0, maint_data.shape[0] - 1):
        # assign group numbers for the timespans in between one maintenance ending and the beginning of the next
        # maintenance time itself remains np.nan assigned
        drift_frame.loc[
            maint_data.values[k] : pd.Timestamp(maint_data.index[k + 1]), "drift_group"
        ] = k

    # define target values for correction
    drift_grouper = drift_frame.groupby("drift_group")
    shift_targets = drift_grouper.aggregate(lambda x: x[:cal_range].mean()).shift(-1)

    for k, group in drift_grouper:
        data_series = group[to_correct.name]
        data_fit, data_shiftTarget = _driftFit(
            data_series, shift_targets.loc[k, :][0], cal_range, model
        )
        data_fit = pd.Series(data_fit, index=group.index)
        data_shiftTarget = pd.Series(data_shiftTarget, index=group.index)
        data_shiftVektor = data_shiftTarget - data_fit
        shiftedData = data_series + data_shiftVektor
        to_correct[shiftedData.index] = shiftedData

    data[field] = to_correct

    return data, flags


@register(mask=["field", "cluster_field"], demask=["cluster_field"], squeeze=[])
def correctRegimeAnomaly(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    cluster_field: str,
    model: CurveFitter,
    tolerance: Optional[str] = None,
    epoch: bool = False,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    flags : saqc.Flags
        Container to store flags of the data.
    cluster_field : str
        A string denoting the field in data, holding the cluster label for the data you want to correct.
    model : Callable
        The model function to be fitted to the regimes.
        It must be a function of the form :math:`f(x, *p)`, where :math:`x` is the ``numpy.array`` holding the
        independent variables and :math:`p` are the model parameters that are to be obtained by fitting.
        Depending on the `x_date` parameter, independent variable x will either be the timestamps
        of every regime transformed to seconds from epoch, or it will be just seconds, counting the regimes length.
    tolerance : {None, str}, default None:
        If an offset string is passed, a data chunk of length `offset` right at the
        start and right at the end is ignored when fitting the model. This is to account for the
        unreliability of data near the changepoints of regimes.
    epoch : bool, default False
        If True, use "seconds from epoch" as x input to the model func, instead of "seconds from regime start".

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    cluster_ser = data[cluster_field]
    unique_successive = pd.unique(cluster_ser.values)
    data_ser = data[field]
    regimes = data_ser.groupby(cluster_ser)
    para_dict = {}
    x_dict = {}
    x_mask = {}
    if tolerance is not None:
        # get seconds
        tolerance = pd.Timedelta(tolerance).total_seconds()
    for label, regime in regimes:
        if epoch is False:
            # get seconds data:
            xdata = (regime.index - regime.index[0]).to_numpy(dtype=float) * 10 ** (-9)
        else:
            # get seconds from epoch data
            xdata = regime.index.to_numpy(dtype=float) * 10 ** (-9)
        ydata = regime.values
        valid_mask = ~np.isnan(ydata)
        if tolerance is not None:
            valid_mask &= xdata > xdata[0] + tolerance
            valid_mask &= xdata < xdata[-1] - tolerance
        try:
            p, *_ = curve_fit(model, xdata[valid_mask], ydata[valid_mask])
        except (RuntimeError, ValueError):
            p = np.array([np.nan])
        para_dict[label] = p
        x_dict[label] = xdata
        x_mask[label] = valid_mask

    first_normal = unique_successive > 0
    first_valid = np.array(
        [
            ~pd.isna(para_dict[unique_successive[i]]).any()
            for i in range(0, unique_successive.shape[0])
        ]
    )
    first_valid = np.where(first_normal & first_valid)[0][0]
    last_valid = 1

    for k in range(0, unique_successive.shape[0]):
        if unique_successive[k] < 0 & (
            not pd.isna(para_dict[unique_successive[k]]).any()
        ):
            ydata = data_ser[regimes.groups[unique_successive[k]]].values
            xdata = x_dict[unique_successive[k]]
            ypara = para_dict[unique_successive[k]]
            if k > 0:
                target_para = para_dict[unique_successive[k - last_valid]]
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
    return data, flags


@register(mask=["field"], demask=[], squeeze=[])
def correctOffset(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    max_jump: float,
    spread: float,
    window: str,
    min_periods: int,
    tolerance: Optional[str] = None,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flags : saqc.Flags
        Container to store flags of the data.
    max_jump : float
        when searching for changepoints in mean - this is the threshold a mean difference in the
        sliding window search must exceed to trigger changepoint detection.
    spread : float
        threshold denoting the maximum, regimes are allowed to abolutely differ in their means
        to form the "normal group" of values.
    window : str
        Size of the adjacent windows that are used to search for the mean changepoints.
    min_periods : int
        Minimum number of periods a search window has to contain, for the result of the changepoint
        detection to be considered valid.
    tolerance : {None, str}, default None:
        If an offset string is passed, a data chunk of length `offset` right from the
        start and right before the end of any regime is ignored when calculating a regimes mean for data correcture.
        This is to account for the unrelyability of data near the changepoints of regimes.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    data, flags = copyField(data, field, flags, field + "_CPcluster")
    data, flags = _assignChangePointCluster(
        data,
        field + "_CPcluster",
        flags,
        lambda x, y: np.abs(np.mean(x) - np.mean(y)),
        lambda x, y: max_jump,
        window=window,
        min_periods=min_periods,
    )
    data, flags = _assignRegimeAnomaly(data, field, flags, field + "_CPcluster", spread)
    data, flags = correctRegimeAnomaly(
        data,
        field,
        flags,
        field + "_CPcluster",
        lambda x, p1: np.array([p1] * x.shape[0]),
        tolerance=tolerance,
    )
    data, flags = dropField(data, field + "_CPcluster", flags)

    return data, flags


def _driftFit(x, shift_target, cal_mean, driftModel):
    x_index = x.index - x.index[0]
    x_data = x_index.total_seconds().values
    x_data = x_data / x_data[-1]
    y_data = x.values
    origin_mean = np.mean(y_data[:cal_mean])
    target_mean = np.mean(y_data[-cal_mean:])

    dataFitFunc = functools.partial(driftModel, origin=origin_mean, target=target_mean)
    # if drift model has free parameters:
    try:
        # try fitting free parameters
        fit_paras, *_ = curve_fit(dataFitFunc, x_data, y_data)
        data_fit = dataFitFunc(x_data, *fit_paras)
        data_shift = driftModel(
            x_data, *fit_paras, origin=origin_mean, target=shift_target
        )
    except RuntimeError:
        # if fit fails -> make no correction
        data_fit = np.array([0] * len(x_data))
        data_shift = np.array([0] * len(x_data))
    # when there are no free parameters in the model:
    except ValueError:
        data_fit = dataFitFunc(x_data)
        data_shift = driftModel(x_data, origin=origin_mean, target=shift_target)

    return data_fit, data_shift


@flagging()
def flagRegimeAnomaly(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    cluster_field: str,
    spread: float,
    method: LinkageString = "single",
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.abs(
        np.nanmean(x) - np.nanmean(y)
    ),
    frac: float = 0.5,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Flags anomalous regimes regarding to modelling regimes of field.

    "Normality" is determined in terms of a maximum spreading distance,
    regimes must not exceed in respect to a certain metric and linkage method.

    In addition, only a range of regimes is considered "normal", if it models
    more then `frac` percentage of the valid samples in "field".

    Note, that you must detect the regime changepoints prior to calling this function.

    Note, that it is possible to perform hypothesis tests for regime equality
    by passing the metric a function for p-value calculation and selecting linkage
    method "complete".

    Parameters
    ----------
    data : dios.DictOfSeries
        Data to process
    field : str
        Name of the column to process
    flags : saqc.Flags
        Container to store flags of the data.
    cluster_field : str
        Column in data, holding the cluster labels for the samples in field.
        (has to be indexed equal to field)
    spread : float
        A threshold denoting the value level, up to wich clusters a agglomerated.
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method for hierarchical (agglomerative) clustering of the variables.
    metric : Callable, default lambda x,y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes.
        Defaults to the difference in mean.
    frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.
    flag : float, default BAD
        flag to set.

    Returns
    -------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The flags object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flags input.
    """
    reserverd = ["set_cluster", "set_flags"]
    kwargs = filterKwargs(kwargs, reserverd)
    return _assignRegimeAnomaly(
        data=data,
        field=field,
        flags=flags,
        cluster_field=cluster_field,
        spread=spread,
        method=method,
        metric=metric,
        frac=frac,
        flag=flag,
        **kwargs,
        set_cluster=False,
        set_flags=True,
    )


@register(mask=["field", "cluster_field"], demask=["cluster_field"], squeeze=[])
def assignRegimeAnomaly(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    cluster_field: str,
    spread: float,
    method: LinkageString = "single",
    metric: Callable[[np.array, np.array], float] = lambda x, y: np.abs(
        np.nanmean(x) - np.nanmean(y)
    ),
    frac: float = 0.5,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    A function to detect values belonging to an anomalous regime regarding modelling
    regimes of field.

    The function changes the value of the regime cluster labels to be negative.
    "Normality" is determined in terms of a maximum spreading distance, regimes must
    not exceed in respect to a certain metric and linkage method. In addition,
    only a range of regimes is considered "normal", if it models more then `frac`
    percentage of the valid samples in "field". Note, that you must detect the regime
    changepoints prior to calling this function. (They are expected to be stored
    parameter `cluster_field`.)

    Note, that it is possible to perform hypothesis tests for regime equality by
    passing the metric a function for p-value calculation and selecting linkage
    method "complete".

    Parameters
    ----------
    data : dios.DictOfSeries
        Data to process
    field : str
        Name of the column to process
    flags : saqc.Flags
        Container to store flags of the data.
    cluster_field : str
        Column in data, holding the cluster labels for the samples in field.
        (has to be indexed equal to field)
    spread : float
        A threshold denoting the value level, up to wich clusters a agglomerated.
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method for hierarchical (agglomerative) clustering of the variables.
    metric : Callable, default lambda x,y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes.
        Defaults to the difference in mean.
    frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The flags object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flags input.
    """
    reserverd = ["set_cluster", "set_flags", "flag"]
    kwargs = filterKwargs(kwargs, reserverd)
    return _assignRegimeAnomaly(
        data=data,
        field=field,
        flags=flags,
        cluster_field=cluster_field,
        spread=spread,
        method=method,
        metric=metric,
        frac=frac,
        **kwargs,
        # control args
        set_cluster=True,
        set_flags=False,
    )


def _assignRegimeAnomaly(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    cluster_field: str,
    spread: float,
    method: LinkageString = "single",
    metric: Callable[[np.array, np.array], float] = lambda x, y: np.abs(
        np.nanmean(x) - np.nanmean(y)
    ),
    frac: float = 0.5,
    set_cluster: bool = True,
    set_flags: bool = False,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    series = data[cluster_field]
    cluster = np.unique(series)
    cluster_dios = DictOfSeries({i: data[field][series == i] for i in cluster})
    plateaus = detectDeviants(cluster_dios, metric, spread, frac, method, "samples")

    if set_flags:
        for p in plateaus:
            flags[cluster_dios.iloc[:, p].index, field] = flag

    if set_cluster:
        for p in plateaus:
            if cluster[p] > 0:
                series[series == cluster[p]] = -cluster[p]

    data[cluster_field] = series
    return data, flags
