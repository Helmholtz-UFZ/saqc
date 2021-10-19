#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Tuple, Callable, Sequence, Optional
from typing_extensions import Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import flagging, Flags
from saqc.lib.tools import toSequence
import saqc.lib.ts_operators as ts_ops


@flagging(masking="all")
def assignKNNScore(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    fields: Sequence[str],
    n: int = 10,
    trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
    trafo_on_partition: bool = True,
    func: Callable[[pd.Series], float] = np.sum,
    target: str = "kNN_scores",
    freq: Union[float, str] = np.inf,
    min_periods: int = 2,
    method: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
    metric: str = "minkowski",
    p: int = 2,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    TODO: docstring need a rework
    Score datapoints by an aggregation of the dictances to their k nearest neighbors.

    The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).

    The steps taken to calculate the scores are as follows:

    1. All the timeseries, named fields, are combined to one feature space by an *inner* join on their date time indexes.
       thus, only samples, that share timestamps across all fields will be included in the feature space
    2. Any datapoint/sample, where one ore more of the features is invalid (=np.nan) will get excluded.
    3. For every data point, the distance to its `n` nearest neighbors is calculated by applying the
       metric `metric` at grade `p` onto the feature space. The defaults lead to the euclidian to be applied.
       If `radius` is not None, it sets the upper bound of distance for a neighbor to be considered one of the
       `n` nearest neighbors. Furthermore, the `freq` argument determines wich samples can be
       included into a datapoints nearest neighbors list, by segmenting the data into chunks of specified temporal
       extension and feeding that chunks to the kNN algorithm seperatly.
    4. For every datapoint, the calculated nearest neighbors distances get aggregated to a score, by the function
       passed to `func`. The default, ``sum`` obviously just sums up the distances.
    5. The resulting timeseries of scores gets assigned to the field target.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flags : saqc.flags
        A flags object, holding flags and additional informations related to `data`.fields
    n : int, default 10
        The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
    trafo : Callable[np.array, np.array], default lambda x: x
        Transformation to apply on the variables before kNN scoring
    trafo_on_partition : bool, default True
        Weather or not to apply the transformation `trafo` onto the whole variable or onto each partition seperatly.
    func : Callable[numpy.array, float], default np.sum
        A function that assigns a score to every one dimensional array, containing the distances
        to every datapoints `n` nearest neighbors.
    target : str, default 'kNN_scores'
        Name of the field, where the resulting scores should be written to.
    freq : {np.inf, float, str}, default np.inf
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.

        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
          string

    min_periods : int, default 2
        The minimum number of periods that have to be present in a partition for the kNN scoring
        to be applied. If the number of periods present is below `min_periods`, the score for the
        datapoints in that partition will be np.nan.
    method : {'ball_tree', 'kd_tree', 'brute', 'auto'}, default 'ball_tree'
        The search algorithm to find each datapoints k nearest neighbors.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    metric : str, default 'minkowski'
        The metric the distances to any datapoints neighbors is computed with. The default of `metric`
        together with the default of `p` result in the euclidian to be applied.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    p : int, default 2
        The grade of the metrice specified by parameter `metric`.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    radius : {None, float}, default None
        If the radius is not None, only the distance to neighbors that ly within the range specified by `radius`
        are comprised in the scoring aggregation.
        The scoring method passed must be capable of handling np.nan values - since, for every point missing
        within `radius` range to make complete the list of the distances to the `n` nearest neighbors,
        one np.nan value gets appended to the list passed to the scoring method.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    data = data.copy()
    fields = toSequence(fields)

    val_frame = data[fields]
    score_index = val_frame.index_of("shared")
    score_ser = pd.Series(np.nan, index=score_index, name=target)

    val_frame = val_frame.loc[val_frame.index_of("shared")].to_df()
    val_frame.dropna(inplace=True)

    if not trafo_on_partition:
        val_frame = val_frame.transform(trafo)

    if val_frame.empty:
        flags[:, field] = UNTOUCHED
        return data, flags

    # partitioning
    if not freq:
        freq = val_frame.shape[0]

    if isinstance(freq, str):
        grouper = pd.Grouper(freq=freq)
    else:
        grouper = pd.Series(
            data=np.arange(0, val_frame.shape[0]), index=val_frame.index
        )
        grouper = grouper.transform(lambda x: int(np.floor(x / freq)))

    partitions = val_frame.groupby(grouper)

    for _, partition in partitions:
        if partition.empty or (partition.shape[0] < min_periods):
            continue
        if trafo_on_partition:
            partition = partition.transform(trafo)
            partition.dropna(inplace=True)

        sample_size = partition.shape[0]
        nn_neighbors = min(n - 1, max(sample_size, 2))
        dist, *_ = ts_ops.kNN(
            partition.values, nn_neighbors, algorithm=method, metric=metric, p=p
        )
        try:
            resids = getattr(dist, func.__name__)(axis=1)
        except AttributeError:
            resids = np.apply_along_axis(func, 1, dist)

        score_ser[partition.index] = resids

    # TODO: this unconditionally overwrite a column, may we should fire a warning ? -- palmb
    if target in flags.columns:
        flags.drop(target)
    flags[target] = pd.Series(UNFLAGGED, index=score_ser.index, dtype=float)

    data[target] = score_ser

    return data, flags
