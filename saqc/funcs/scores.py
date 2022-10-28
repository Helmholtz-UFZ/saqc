#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Literal

import saqc.lib.ts_operators as ts_ops
from saqc.constants import UNFLAGGED
from saqc.core.register import register
from saqc.lib.tools import getApply, toSequence

if TYPE_CHECKING:
    from saqc.core.core import SaQC


def _univarScoring(
    data: pd.Series,
    window: Optional[str, int] = None,
    norm_func: Callable = np.nanstd,
    model_func: Callable = np.nanmean,
    center: bool = True,
    min_periods: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate (rolling) normalisation scores.

    Parameters
    ----------
    data
        A dictionary of pandas.Series, holding all the data.
    window : {str, int}, default None
            Size of the window. Either determined via an Offset String, denoting the windows temporal extension or
            by an integer, denoting the windows number of periods.
            `NaN` measurements also count as periods.
            If `None` is passed, All data points share the same scoring window, which than equals the whole
            data.
    model_func
        Function to calculate the center moment in every window.
    norm_func
        Function to calculate the scaling for every window
    center
        Weather or not to center the target value in the scoring window. If `False`, the
        target value is the last value in the window.
    min_periods
        Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
    """
    if data.empty:
        return data, data, data
    if min_periods is None:
        min_periods = 0

    if window is None:
        if data.notna().sum() >= min_periods:
            # in case of stationary analysis, broadcast statistics to series for compatibility reasons
            norm = pd.Series(norm_func(data.values), index=data.index)
            model = pd.Series(model_func(data.values), index=data.index)
        else:
            norm = pd.Series(np.nan, index=data.index)
            model = pd.Series(np.nan, index=data.index)
    else:
        # wrap passed func with rolling built in if possible and rolling.apply else
        roller = data.rolling(window=window, min_periods=min_periods, center=center)
        norm = getApply(roller, norm_func)
        model = getApply(roller, model_func)

    score = (data - model) / norm
    return score, model, norm


class ScoresMixin:
    @register(
        mask=["field"],
        demask=[],
        squeeze=["target"],
        multivariate=True,
        handles_target=True,
    )
    def assignKNNScore(
        self: "SaQC",
        field: Sequence[str],
        target: str,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        freq: float | str | None = np.inf,
        min_periods: int = 2,
        method: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        metric: str = "minkowski",
        p: int = 2,
        **kwargs,
    ) -> "SaQC":
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
        field : list of str
            input variable names.

        target : str, default "kNNscores"
            A new Column name, where the result is stored.

        n : int, default 10
            The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.

        func : Callable[numpy.array, float], default np.sum
            A function that assigns a score to every one dimensional array, containing the distances
            to every datapoints `n` nearest neighbors.

        freq : {float, str, None}, default np.inf
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

        Returns
        -------
        saqc.SaQC

        References
        ----------
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        if isinstance(target, list):
            if (len(target) > 1) or (target[0] in self._data.columns):
                raise ValueError(
                    f"'target' must not exist and be of length 1. {target} was passed instead."
                )
            target = target[0]

        fields = toSequence(field)
        val_frame = self._data[fields].copy()
        score_index = val_frame.index_of("shared")
        score_ser = pd.Series(np.nan, index=score_index, name=target)

        val_frame = val_frame.loc[val_frame.index_of("shared")].to_df()
        val_frame.dropna(inplace=True)

        if val_frame.empty:
            return self

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

        self._flags[target] = pd.Series(UNFLAGGED, index=score_ser.index, dtype=float)
        self._data[target] = score_ser

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def assignZScore(
        self: "SaQC",
        field: str,
        window: Optional[str] = None,
        norm_func: Callable = np.nanstd,
        model_func: Callable = np.nanmean,
        center: bool = True,
        min_periods: Optional[int] = None,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate (rolling) Zscores.

        See the Notes section for a detailed overview of the calculation

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
        window : {str, int}, default None
            Size of the window. Either determined via an Offset String, denoting the windows temporal extension or
            by an integer, denoting the windows number of periods.
            `NaN` measurements also count as periods.
            If `None` is passed, All data points share the same scoring window, which than equals the whole
            data.
        model_func
            Function to calculate the center moment in every window.
        norm_func
            Function to calculate the scaling for every window
        center
            Weather or not to center the target value in the scoring window. If `False`, the
            target value is the last value in the window.
        min_periods
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        flags : saqc.Flags
            The quality flags of data
            Flags values may have changed, relatively to the flags input.

        Notes
        -----
        Steps of calculation:

        1. Consider a window :math:`W` of successive points :math:`W = x_{1},...x_{w}`
        containing the value :math:`y_{K}` wich is to be checked.
        (The index of :math:`K` depends on the selection of the parameter `center`.)

        2. The "moment" :math:`M` for the window gets calculated via :math:`M=` `model_func(:math:`W`)

        3. The "scaling" :math:`N` for the window gets calculated via :math:`N=` `norm_func(:math:`W`)

        4. The "score" :math:`S` for the point :math:`x_{k}`gets calculated via :math:`S=(x_{k} - M) / N`
        """

        if min_periods is None:
            min_periods = 0

        score, _, _ = _univarScoring(
            self._data[field],
            window=window,
            norm_func=norm_func,
            model_func=model_func,
            center=center,
            min_periods=min_periods,
        )
        self._data[field] = score
        return self
