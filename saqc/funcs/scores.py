#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Tuple, Callable, Sequence, Optional
from typing_extensions import Literal

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.common import *
from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.lib import ts_operators as ts_ops
from saqc.lib.tools import toSequence


@register(masking='all', module="scores")
def assignKNNScore(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        fields: Sequence[str],
        n_neighbors: int=10,
        trafo: Callable[[pd.Series], pd.Series]=lambda x: x,
        trafo_on_partition: bool=True,
        scoring_func: Callable[[pd.Series], float]=np.sum,
        target_field: str='kNN_scores',
        partition_freq: Union[float, str]=np.inf,
        partition_min: int=2,
        kNN_algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"]='ball_tree',
        metric: str='minkowski',
        p: int=2,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Score datapoints by an aggregation of the dictances to their k nearest neighbors.

    The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).

    The steps taken to calculate the scores are as follows:

    1. All the timeseries, named fields, are combined to one feature space by an *inner* join on their date time indexes.
       thus, only samples, that share timestamps across all fields will be included in the feature space
    2. Any datapoint/sample, where one ore more of the features is invalid (=np.nan) will get excluded.
    3. For every data point, the distance to its `n_neighbors` nearest neighbors is calculated by applying the
       metric `metric` at grade `p` onto the feature space. The defaults lead to the euclidian to be applied.
       If `radius` is not None, it sets the upper bound of distance for a neighbor to be considered one of the
       `n_neigbors` nearest neighbors. Furthermore, the `partition_freq` argument determines wich samples can be
       included into a datapoints nearest neighbors list, by segmenting the data into chunks of specified temporal
       extension and feeding that chunks to the kNN algorithm seperatly.
    4. For every datapoint, the calculated nearest neighbors distances get aggregated to a score, by the function
       passed to `scoring_func`. The default, ``sum`` obviously just sums up the distances.
    5. The resulting timeseries of scores gets assigned to the field target_field.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.fields
    n_neighbors : int, default 10
        The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
    trafo : Callable[np.array, np.array], default lambda x: x
        Transformation to apply on the variables before kNN scoring
    trafo_on_partition : bool, default True
        Weather or not to apply the transformation `trafo` onto the whole variable or onto each partition seperatly.
    scoring_func : Callable[numpy.array, float], default np.sum
        A function that assigns a score to every one dimensional array, containing the distances
        to every datapoints `n_neighbors` nearest neighbors.
    target_field : str, default 'kNN_scores'
        Name of the field, where the resulting scores should be written to.
    partition_freq : {np.inf, float, str}, default np.inf
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.

        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
          string

    partition_min : int, default 2
        The minimum number of periods that have to be present in a partition for the kNN scoring
        to be applied. If the number of periods present is below `partition_min`, the score for the
        datapoints in that partition will be np.nan.
    kNN_algorithm : {'ball_tree', 'kd_tree', 'brute', 'auto'}, default 'ball_tree'
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
        within `radius` range to make complete the list of the distances to the `n_neighbors` nearest neighbors,
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
    score_ser = pd.Series(np.nan, index=score_index, name=target_field)

    val_frame = val_frame.loc[val_frame.index_of("shared")].to_df()
    val_frame.dropna(inplace=True)

    if not trafo_on_partition:
        val_frame = val_frame.transform(trafo)

    if val_frame.empty:
        return data, flagger

    # partitioning
    if not partition_freq:
        partition_freq = val_frame.shape[0]

    if isinstance(partition_freq, str):
        grouper = pd.Grouper(freq=partition_freq)
    else:
        grouper = pd.Series(data=np.arange(0, val_frame.shape[0]), index=val_frame.index)
        grouper = grouper.transform(lambda x: int(np.floor(x / partition_freq)))

    partitions = val_frame.groupby(grouper)

    for _, partition in partitions:
        if partition.empty or (partition.shape[0] < partition_min):
            continue
        if trafo_on_partition:
            partition = partition.transform(trafo)
            partition.dropna(inplace=True)

        sample_size = partition.shape[0]
        nn_neighbors = min(n_neighbors - 1, max(sample_size, 2))
        dist, *_ = ts_ops.kNN(partition.values, nn_neighbors, algorithm=kNN_algorithm, metric=metric, p=p)
        try:
            resids = getattr(dist, scoring_func.__name__)(axis=1)
        except AttributeError:
            resids = np.apply_along_axis(scoring_func, 1, dist)

        score_ser[partition.index] = resids

    # TODO: this unconditionally overwrite a column, may we should fire a warning ? -- palmb
    if target_field in flagger.columns:
        flagger.drop(target_field)
    flagger[target_field] = pd.Series(UNFLAGGED, index=score_ser.index, dtype=float)

    data[target_field] = score_ser
    return data, flagger
