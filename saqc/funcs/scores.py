#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Tuple, Callable, Sequence, Optional
from typing_extensions import Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import register, Flags
from saqc.lib.tools import toSequence
import saqc.lib.ts_operators as ts_ops


@register(
    mask=["field"],
    demask=[],
    squeeze=["target"],
    multivariate=True,
    handles_target=True,
)
def assignKNNScore(
    data: DictOfSeries,
    field: Sequence[str],
    flags: Flags,
    target: str,
    n: int = 10,
    func: Callable[[pd.Series], float] = np.sum,
    freq: Union[float, str] = np.inf,
    min_periods: int = 2,
    method: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
    metric: str = "minkowski",
    p: int = 2,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    TODO: docstring need a rework
    Score datapoints by an aggregation of the dictances to their k nearest neighbors.

    The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).

    The steps taken to calculate the scores are as follows:

    1. All the timeseries, given through ``field``, are combined to one feature space by an *inner* join on their
       date time indexes. thus, only samples, that share timestamps across all ``field`` will be included in the
       feature space.
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
    field : list of str
        input variable names.
    flags : saqc.flags
        A flags object, holding flags and additional informations related to `data`.
    target : str, default "kNNscores"
        A new Column name, where the result is stored.
    n : int, default 10
        The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
    func : Callable[numpy.array, float], default np.sum
        A function that assigns a score to every one dimensional array, containing the distances
        to every datapoints `n` nearest neighbors.
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

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    if isinstance(target, list):
        if (len(target) > 1) or (target[0] in data.columns):
            raise ValueError(
                f"'target' must not exist and be of length 1. {target} was passed instead."
            )
        target = target[0]

    fields = toSequence(field)
    val_frame = data[fields].copy()
    score_index = val_frame.index_of("shared")
    score_ser = pd.Series(np.nan, index=score_index, name=target)

    val_frame = val_frame.loc[val_frame.index_of("shared")].to_df()
    val_frame.dropna(inplace=True)

    if val_frame.empty:
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

        sample_size = partition.shape[0]
        nn_neighbors = min(n, max(sample_size, 2) - 1)
        dist, *_ = ts_ops.kNN(
            partition.values, nn_neighbors, algorithm=method, metric=metric, p=p
        )
        try:
            resids = getattr(dist, func.__name__)(axis=1)
        except AttributeError:
            resids = np.apply_along_axis(func, 1, dist)

        score_ser[partition.index] = resids

    flags[target] = pd.Series(UNFLAGGED, index=score_ser.index, dtype=float)
    data[target] = score_ser

    return data, flags
