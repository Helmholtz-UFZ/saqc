#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from typing_extensions import Literal

from saqc import UNFLAGGED
from saqc.core import register
from saqc.lib.checking import (
    validateChoice,
    validateFuncSelection,
    validateMinPeriods,
    validateWindow,
)
from saqc.lib.docs import DOC_TEMPLATES
from saqc.lib.tools import getApply, toSequence
from saqc.lib.ts_operators import kNN
from saqc.parsing.environ import ENV_OPERATORS, ENV_TRAFOS

if TYPE_CHECKING:
    from saqc import SaQC


def _kNNApply(vals, n_neighbors, func=np.sum, **kwargs):
    dist, *_ = kNN(vals, n_neighbors=n_neighbors, **kwargs)
    try:
        resids = getattr(dist, func.__name__)(axis=1)
    except AttributeError:
        resids = np.apply_along_axis(func, 1, dist)
    return resids


def _LOFApply(vals, n_neighbors, ret="scores", **kwargs):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, **kwargs)
    labels = clf.fit_predict(vals)
    resids = clf.negative_outlier_factor_
    if ret == "scores":
        return resids
    else:
        return labels == -1


def _groupedScoring(
    val_frame: pd.DataFrame,
    n: int = 20,
    freq: float | str | None = np.inf,
    min_periods: int = 2,
    score_func: Callable = _LOFApply,
    score_kwargs: Optional[dict] = None,
) -> pd.Series:
    score_kwargs = score_kwargs or {}
    score_index = val_frame.index
    score_ser = pd.Series(np.nan, index=score_index)

    val_frame = val_frame.loc[score_index]
    val_frame = val_frame.dropna()

    if val_frame.empty:
        return score_ser

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
        resids = score_func(partition.values, nn_neighbors, **score_kwargs)
        score_ser[partition.index] = resids

    return score_ser


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
    data :
        A dictionary of pandas.Series, holding all the data.
    window :
            Size of the window. Either determined via an Offset String, denoting the windows temporal extension or
            by an integer, denoting the windows number of periods.
            `NaN` measurements also count as periods.
            If `None` is passed, All data points share the same scoring window, which than equals the whole
            data.
    model_func : default std
        Function to calculate the center moment in every window.
    norm_func : default mean
        Function to calculate the scaling for every window
    center
        Weather or not to center the target value in the scoring window. If `False`, the
        target value is the last value in the window.
    min_periods
        Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
    """
    validateFuncSelection(model_func, "model_func", allow_operator_str=True)
    if isinstance(model_func, str):
        model_func = ENV_OPERATORS[model_func]
    validateFuncSelection(norm_func, "norm_func", allow_operator_str=True)
    if isinstance(norm_func, str):
        norm_func = ENV_OPERATORS[norm_func]
    validateWindow(window, optional=True)
    validateMinPeriods(min_periods, optional=True)

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
        docstring={"field": DOC_TEMPLATES["field"], "target": DOC_TEMPLATES["target"]},
    )
    def assignKNNScore(
        self: "SaQC",
        field: Sequence[str],
        target: str,
        n: int = 10,
        func: Callable[[pd.Series], float] | str = "sum",
        freq: float | str | None = np.inf,
        min_periods: int = 2,
        algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        metric: str = "minkowski",
        p: int = 2,
        **kwargs,
    ) -> "SaQC":
        """
        Score datapoints by an aggregation of the distances to their `k` nearest neighbors.

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
        n : :
            The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.

        func : default sum
            A function that assigns a score to every one dimensional array, containing the distances
            to every datapoints `n` nearest neighbors.

        freq :
            Determines the segmentation of the data into partitions, the kNN algorithm is
            applied onto individually.

            * ``np.inf``: Apply Scoring on whole data set at once
            * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
            * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
              string

        min_periods :
            The minimum number of periods that have to be present in a window for the kNN scoring
            to be applied. If the number of periods present is below `min_periods`, the score for the
            datapoints in that window will be np.nan.

        algorithm :
            The search algorithm to find each datapoints k nearest neighbors.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.

        metric :
            The metric the distances to any datapoints neighbors is computed with. The default of `metric`
            together with the default of `p` result in the euclidian to be applied.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.

        p : :
            The grade of the metrice specified by parameter `metric`.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.

        References
        ----------
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        validateFuncSelection(func, allow_operator_str=True)
        if isinstance(func, str):
            func = ENV_OPERATORS[func]
        validateChoice(
            algorithm, "algorithm", ["ball_tree", "kd_tree", "brute", "auto"]
        )

        if isinstance(target, list):
            if len(target) > 1:
                raise ValueError(
                    f"'target' must be of length 1. {target} was passed instead."
                )
            target = target[0]
        if target in self._data.columns:
            self = self.dropField(target)

        fields = toSequence(field)
        val_frame = self._data[fields].copy().to_pandas()
        score_ser = _groupedScoring(
            val_frame,
            n=n,
            freq=freq,
            min_periods=min_periods,
            score_func=_kNNApply,
            score_kwargs={
                "func": func,
                "metric": metric,
                "p": p,
                "algorithm": algorithm,
            },
        )
        score_ser.name = target

        self._flags[target] = pd.Series(np.nan, index=score_ser.index, dtype=float)
        self._data[target] = score_ser

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def assignZScore(
        self: "SaQC",
        field: str,
        window: str | None = None,
        norm_func: Callable | str = "std",
        model_func: Callable | str = "mean",
        center: bool = True,
        min_periods: int | None = None,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate (rolling) Zscores.

        See the Notes section for a detailed overview of the calculation

        Parameters
        ----------
        window :
            Size of the window. can be determined as:
            * Offset String, denoting the windows temporal extension
            * Integer, denoting the windows number of periods.
            * `None` (default), All data points share the same scoring window, which than equals the whole
            data.
        model_func : default std
            Function to calculate the center moment in every window.
        norm_func : default mean
            Function to calculate the scaling for every window
        center :
            Weather or not to center the target value in the scoring window. If `False`, the
            target value is the last value in the window.
        min_periods :
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.

        Notes
        -----
        Steps of calculation:

        1. Consider a window :math:`W` of successive points :math:`W = x_{1},...x_{w}`
        containing the value :math:`y_{K}` wich is to be checked.
        (The index of :math:`K` depends on the selection of the parameter `center`.)

        2. The "moment" :math:`M` for the window gets calculated via :math:`M=` model_func(:math:`W`)

        3. The "scaling" :math:`N` for the window gets calculated via :math:`N=` norm_func(:math:`W`)

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

    @register(
        mask=["field"],
        demask=[],
        squeeze=["target"],
        multivariate=True,
        handles_target=True,
        docstring={"field": DOC_TEMPLATES["field"], "target": DOC_TEMPLATES["target"]},
    )
    def assignLOF(
        self: "SaQC",
        field: Sequence[str],
        target: str,
        n: int = 20,
        freq: float | str | None = np.inf,
        min_periods: int = 2,
        algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        p: int = 2,
        **kwargs,
    ) -> "SaQC":
        """
        Assign Local Outlier Factor (LOF).

        Parameters
        ----------
        n :
            Number of periods to be included into the LOF calculation. Defaults to `20`, which is a value found to be
            suitable in the literature.
        freq :
            Determines the segmentation of the data into partitions, the kNN algorithm is
            applied onto individually.
        algorithm :
            Algorithm used for calculating the `n`-nearest neighbors needed for LOF calculation.
        p :
            Degree of the metric ("Minkowski"), according to wich distance to neighbors is determined.
            Most important values are:

            * `1` - Manhatten Metric
            * `2` - Euclidian Metric

        Notes
        -----

        * `n` determines the "locality" of an observation (its `n` nearest neighbors) and sets the upper limit of
          values of an outlier clusters (i.e. consecutive outliers). Outlier clusters of size greater than `n/2`
          may not be detected reliably.
        * The larger `n`, the lesser the algorithm's sensitivity to local outliers and small or singleton outliers
          points. Higher values greatly increase numerical costs.
        """
        from saqc.funcs.outliers import OutliersMixin

        validateMinPeriods(min_periods)
        OutliersMixin._validateLOF(algorithm, n, p, 1.0)

        if isinstance(target, list):
            if len(target) > 1:
                raise ValueError(
                    f"'target' must be of length 1. {target} was passed instead."
                )
            target = target[0]
        if target in self._data.columns:
            self = self.dropField(target)

        fields = toSequence(field)
        val_frame = self._data[fields].copy().to_pandas()

        score_ser = _groupedScoring(
            val_frame,
            n=n,
            freq=freq,
            min_periods=min_periods,
            score_func=_LOFApply,
            score_kwargs={
                "metric": "minkowski",
                "contamination": "auto",
                "p": p,
                "algorithm": algorithm,
            },
        )
        score_ser.name = target
        self._flags[target] = pd.Series(np.nan, index=score_ser.index, dtype=float)
        self._data[target] = score_ser

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def assignUniLOF(
        self: "SaQC",
        field: str,
        n: int = 20,
        algorithm: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        p: int = 1,
        density: Literal["auto"] | float = "auto",
        fill_na: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Assign "univariate" Local Outlier Factor (LOF).

        The Function is a wrapper around a usual LOF implementation, aiming for an easy to use, parameter minimal
        outlier scoring function for singleton variables, that does not necessitate prior modelling of the variable.
        LOF is applied onto a concatenation of the `field` variable and a "temporal density", or "penalty" variable,
        that measures temporal distance between data points.

        See the Notes section for more details on the algorithm.

        Parameters
        ----------
        n :
            Number of periods to be included into the LOF calculation. Defaults to `20`, which is a value found to be
            suitable in the literature.

            * `n` determines the "locality" of an observation (its `n` nearest neighbors) and sets the upper limit of
              values of an outlier clusters (i.e. consecutive outliers). Outlier clusters of size greater than `n/2`
              may not be detected reliably.
            * The larger `n`, the lesser the algorithm's sensitivity to local outliers and small or singleton outliers
              points. Higher values greatly increase numerical costs.

        algorithm :
            Algorithm used for calculating the `n`-nearest neighbors needed for LOF calculation.

        p :
            Degree of the metric ("Minkowski"), according to wich distance to neighbors is determined.
            Most important values are:

            * `1` - Manhatten Metric
            * `2` - Euclidian Metric

        density :
            How to calculate the temporal distance/density for the variable-to-be-flagged.

            * float - introduces linear density with an increment equal to `density`
            * Callable - calculates the density by applying the function passed onto the variable to be flagged
              (passed as Series).

        fill_na :
            If True, NaNs in the data are filled with a linear interpolation.

        Notes
        -----
        Algorithm steps for uniLOF flagging of variable `x`:

        1. The temporal density `dt(x)` is calculated according to the `density` parameter.
        2. LOF scores `LOF(x)` are calculated for the concatenation [`x`, `dt(x)`]
        3. `x` is flagged where `LOF(x)` exceeds the threshold determined by the parameter `thresh`.

        Examples
        --------

        """
        from saqc.funcs.outliers import OutliersMixin

        OutliersMixin._validateLOF(algorithm, n, p, density)

        vals = self._data[field]

        if fill_na:
            filled = vals.isna()
            vals = vals.interpolate("linear")
            filled = filled & vals.notna()
        else:
            filled = pd.Series(False, index=vals.index)

        if density == "auto":
            v_diff = vals.diff()
            density = v_diff.abs().median()
            if density == 0:
                density = v_diff[v_diff != 0].abs().median()
        elif isinstance(density, Callable):
            density = density(vals)
        if isinstance(density, pd.Series):
            density = density.values

        d_var = pd.Series(np.arange(len(vals)) * density, index=vals.index)
        na_bool_ser = vals.isna() | d_var.isna()
        na_idx = na_bool_ser.index[na_bool_ser.values]
        # notna_bool = vals.notna()
        val_no = (~na_bool_ser).sum()
        if 2 < val_no <= n:
            n = val_no - 2
        elif val_no <= 2:
            return self

        d_var = d_var.drop(na_idx, axis=0).values
        vals = vals.drop(na_idx, axis=0).values
        vals_extended = np.array(
            list(vals[::-1][-n:]) + list(vals) + list(vals[::-1][:n])
        )
        d_extended = np.array(
            list(d_var[:n] + (d_var[0] - d_var[n + 1]))
            + list(d_var)
            + list(d_var[-n:] + (d_var[-1] - d_var[-n - 1]))
        )

        LOF_vars = np.array([vals_extended, d_extended]).T
        scores = _LOFApply(LOF_vars, n, p=p, algorithm=algorithm, metric="minkowski")
        scores = scores[n:-n]
        score_ser = pd.Series(scores, index=na_bool_ser.index[~na_bool_ser.values])
        score_ser = score_ser.reindex(na_bool_ser.index)
        score_ser[filled] = np.nan
        self._data[field] = score_ser
        return self
