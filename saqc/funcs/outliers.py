#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from outliers import smirnov_grubbs
from scipy.stats import median_abs_deviation
from typing_extensions import Literal

from dios import DictOfSeries
from saqc.constants import BAD, UNFLAGGED
from saqc.core.flags import Flags
from saqc.core.register import flagging, register
from saqc.funcs.scores import _univarScoring
from saqc.lib.tools import customRoller, getFreqDelta, toSequence

if TYPE_CHECKING:
    from saqc.core.core import SaQC


class OutliersMixin:
    @flagging()
    def flagRange(
        self: "SaQC",
        field: str,
        min: float = -np.inf,
        max: float = np.inf,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Function flags values exceeding the closed interval [`min`, `max`].

        Parameters
        ----------
        field : str
            The field name of the column, holding the data-to-be-flagged.
        min : float
            Lower bound for valid data.
        max : float
            Upper bound for valid data.
        flag : float, default BAD
            flag to set.

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        flags : saqc.Flags
            The quality flags of data
        """

        # using .values is much faster
        datacol = self._data[field].to_numpy()
        mask = (datacol < min) | (datacol > max)
        self._flags[mask, field] = flag
        return self

    @flagging()
    def flagByStray(
        self: "SaQC",
        field: str,
        window: int | str | None = None,
        min_periods: int = 11,
        iter_start: float = 0.5,
        alpha: float = 0.05,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.

        Find more information on the algorithm in References [1].

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        freq : str, int, or None, default None
            Determines the segmentation of the data into partitions, the kNN algorithm is
            applied onto individually.

            * ``np.inf``: Apply Scoring on whole data set at once
            * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
            * Offset String : Apply scoring on successive partitions of temporal extension
              matching the passed offset string

        min_periods : int, default 11
            Minimum number of periods per partition that have to be present for a valid
            outlier dettection to be made in this partition. (Only of effect, if `freq`
            is an integer.) Partition min value must always be greater then the
            nn_neighbors value.

        iter_start : float, default 0.5
            Float in [0,1] that determines which percentage of data is considered
            "normal". 0.5 results in the stray algorithm to search only the upper 50 % of
            the scores for the cut off point. (See reference section for more information)

        alpha : float, default 0.05
            Level of significance by which it is tested, if a score might be drawn from
            another distribution, than the majority of the data.

        flag : float, default BAD
            flag to set.

        Returns
        -------
        saqc.SaQC

        References
        ----------
        [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in
            high dimensional data. arXiv preprint arXiv:1908.04000.
        """
        scores = self._data[field].dropna()

        if scores.empty:
            return self

        if not window:
            window = scores.shape[0]

        if isinstance(window, str):
            partitions = scores.groupby(pd.Grouper(freq=window))

        else:
            grouper_series = pd.Series(
                data=np.arange(0, scores.shape[0]), index=scores.index
            )
            grouper_series = grouper_series.transform(
                lambda x: int(np.floor(x / window))
            )
            partitions = scores.groupby(grouper_series)

        # calculate flags for every partition
        for _, partition in partitions:

            if partition.empty | (partition.shape[0] < min_periods):
                continue

            sample_size = partition.shape[0]

            sorted_i = partition.values.argsort()
            resids = partition.values[sorted_i]
            gaps = np.append(0, np.diff(resids))

            tail_size = int(max(min(50, np.floor(sample_size / 4)), 2))
            tail_indices = np.arange(2, tail_size + 1)

            i_start = int(max(np.floor(sample_size * iter_start), 1) + 1)
            ghat = np.array([np.nan] * sample_size)

            for i in range(i_start - 1, sample_size):
                ghat[i] = sum(
                    (tail_indices / (tail_size - 1)) * gaps[i - tail_indices + 1]
                )

            log_alpha = np.log(1 / alpha)
            for iter_index in range(i_start - 1, sample_size):
                if gaps[iter_index] > log_alpha * ghat[iter_index]:
                    index = partition.index[sorted_i[iter_index:]]
                    self._flags[index, field] = flag
                    break

        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=False,
    )
    def flagMVScores(
        self: "SaQC",
        field: Sequence[str],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        iter_start: float = 0.5,
        partition: Optional[Union[int, str]] = None,
        partition_min: int = 11,
        stray_range: Optional[str] = None,
        drop_flagged: bool = False,  # TODO: still a case ?
        thresh: float = 3.5,
        min_periods: int = 1,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The algorithm implements a 3-step outlier detection procedure for simultaneously
        flagging of higher dimensional data (dimensions > 3).

        In references [1], the procedure is introduced and exemplified with an
        application on hydrological data. See the notes section for an overview over the
        algorithms basic steps.

        Parameters
        ----------
        field : list of str
            List of fieldnames, corresponding to the variables that are to be included
            into the flagging process.

        trafo : callable, default lambda x:x
            Transformation to be applied onto every column before scoring. Will likely
            get deprecated soon. Its better to transform the data in a processing step,
            preceeeding the call to ``flagMVScores``.

        alpha : float, default 0.05
            Level of significance by which it is tested, if an observations score might
            be drawn from another distribution than the majority of the observation.

        n : int, default 10
            Number of neighbors included in the scoring process for every datapoint.

        func : Callable[numpy.array, float], default np.sum
            The function that maps the set of every points k-nearest neighbor distances
            onto a certain scoring.

        iter_start : float, default 0.5
            Float in [0,1] that determines which percentage of data is considered
            "normal". 0.5 results in the threshing algorithm to search only the upper 50
            % of the scores for the cut off point. (See reference section for more
            information)

        partition : {None, str, int}, default None
            Only effective when `threshing` = 'stray'. Determines the size of the data
            partitions, the data is decomposed into. Each partition is checked seperately
            for outliers. If a String is passed, it has to be an offset string and it
            results in partitioning the data into parts of according temporal length. If
            an integer is passed, the data is simply split up into continous chunks of
            `freq` periods. if ``None`` is passed (default), all the data will be tested
            in one run.

        partition_min : int, default 11
            Only effective when `threshing` = 'stray'. Minimum number of periods per
            partition that have to be present for a valid outlier detection to be made in
            this partition. (Only of effect, if `stray_partition` is an integer.)

        partition_trafo : bool, default True
            Whether or not to apply the passed transformation on every partition the
            algorithm is applied on, separately.

        stray_range : {None, str}, default None
            If not None, it is tried to reduce the stray result onto single outlier
            components of the input fields. An offset string, denoting the range of the
            temporal surrounding to include into the MAD testing while trying to reduce
            flags.

        drop_flagged : bool, default False
            Only effective when `range` is not ``None``. Whether or not to drop flagged
            values other than the value under test from the temporal surrounding before
            checking the value with MAD.

        thresh : float, default 3.5
            Only effective when `range` is not ``None``. The `critical` value,
            controlling wheather the MAD score is considered referring to an outlier or
            not. Higher values result in less rigid flagging. The default value is widely
            considered apropriate in the literature.

        min_periods : int, 1
            Only effective when `range` is not ``None``. Minimum number of meassurements
            necessarily present in a reduction interval for reduction actually to be
            performed.

        flag : float, default BAD
            flag to set.

        Returns
        -------
        saqc.SaQC

        Notes
        -----
        The basic steps are:

        1. transforming

        The different data columns are transformed via timeseries transformations to
        (a) make them comparable and
        (b) make outliers more stand out.

        This step is usually subject to a phase of research/try and error. See [1] for more
        details.

        Note, that the data transformation as an built-in step of the algorithm,
        will likely get deprecated soon. Its better to transform the data in a processing
        step, preceeding the multivariate flagging process. Also, by doing so, one gets
        mutch more control and variety in the transformation applied, since the `trafo`
        parameter only allows for application of the same transformation to all of the
        variables involved.

        2. scoring

        Every observation gets assigned a score depending on its k nearest neighbors. See
        the `scoring_method` parameter description for details on the different scoring
        methods. Furthermore [1], [2] may give some insight in the pro and cons of the
        different methods.

        3. threshing

        The gaps between the (greatest) scores are tested for beeing drawn from the same
        distribution as the majority of the scores. If a gap is encountered, that,
        with sufficient significance, can be said to not be drawn from the same
        distribution as the one all the smaller gaps are drawn from, than the observation
        belonging to this gap, and all the observations belonging to gaps larger then
        this gap, get flagged outliers. See description of the `threshing` parameter for
        more details. Although [2] gives a fully detailed overview over the `stray`
        algorithm.
        """

        fields = toSequence(field)

        fields_ = []
        for f in fields:
            field_ = str(uuid.uuid4())
            self = self.copyField(field=f, target=field_)
            self = self.transform(field=field_, func=trafo, freq=partition)
            fields_.append(field_)

        knn_field = str(uuid.uuid4())
        self = self.assignKNNScore(
            field=fields_,
            target=knn_field,
            n=n,
            func=func,
            freq=partition,
            method="ball_tree",
            min_periods=partition_min,
            **kwargs,
        )
        for field_ in fields_:
            self = self.dropField(field_)

        self = self.flagByStray(
            field=knn_field,
            freq=partition,
            min_periods=partition_min,
            iter_start=iter_start,
            alpha=alpha,
            flag=flag,
            **kwargs,
        )

        self._data, self._flags = _evalStrayLabels(
            data=self._data,
            field=knn_field,
            target=fields,
            flags=self._flags,
            reduction_range=stray_range,
            reduction_drop_flagged=drop_flagged,
            reduction_thresh=thresh,
            reduction_min_periods=min_periods,
            flag=flag,
            **kwargs,
        )
        return self.dropField(knn_field)

    @flagging()
    def flagRaise(
        self: "SaQC",
        field: str,
        thresh: float,
        raise_window: str,
        freq: str,
        average_window: Optional[str] = None,
        raise_factor: float = 2.0,
        slope: Optional[float] = None,
        weight: float = 0.8,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The function flags raises and drops in value courses, that exceed a certain threshold
        within a certain timespan.

        The parameter variety of the function is owned to the intriguing
        case of values, that "return" from outlierish or anomalious value levels and
        thus exceed the threshold, while actually being usual values.

        NOTE: the dataset is NOT supposed to be harmonized to a time series with an
        equidistant frequency grid.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        thresh : float
            The threshold, for the total rise (thresh > 0), or total drop (thresh < 0),
            value courses must not exceed within a timespan of length `raise_window`.

        raise_window : str
            An offset string, determining the timespan, the rise/drop thresholding refers
            to. Window is inclusively defined.

        freq : str
            An offset string, determining The frequency, the timeseries to-be-flagged is
            supposed to be sampled at. The window is inclusively defined.

        average_window : {None, str}, default None
            See condition (2) of the description linked in the references. Window is
            inclusively defined. The window defaults to 1.5 times the size of `raise_window`

        raise_factor : float, default 2
            See second condition listed in the notes below.

        slope : {None, float}, default None
            See third condition listed in the notes below.

        weight : float, default 0.8
            See third condition listed in the notes below.

        flag : float, default BAD
            flag to set.

        Returns
        -------
        saqc.SaQC

        Notes
        -----
        The value :math:`x_{k}` of a time series :math:`x` with associated
        timestamps :math:`t_i`, is flagged a raise, if:

        * There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within `raise_window`
          range, so that:

          * :math:`M = |x_k - x_s | >`  `thresh` :math:`> 0`

        * The weighted average :math:`\\mu^{*}` of the values, preceding :math:`x_{k}`
          within `average_window`
          range indicates, that :math:`x_{k}` does not return from an "outlierish" value
          course, meaning that:

          * :math:`x_k > \\mu^* + ( M` / `mean_raise_factor` :math:`)`

        * Additionally, if ``min_slope`` is not `None`, :math:`x_{k}` is checked for being
          sufficiently divergent from its very predecessor :math:`x_{k-1}`, meaning that, it
          is additionally checked if:

          * :math:`x_k - x_{k-1} >` `min_slope`
          * :math:`t_k - t_{k-1} >` `weight` :math:`\\times` `freq`

        """

        # prepare input args
        dataseries = self._data[field].dropna()
        raise_window_td = pd.Timedelta(raise_window)
        freq_dt = pd.Timedelta(freq)
        if slope is not None:
            slope = np.abs(slope)

        if average_window is None:
            average_window = 1.5 * raise_window_td

        if thresh < 0:
            dataseries *= -1
            thresh *= -1

        def raise_check(x, thresh):
            test_set = x[-1] - x[0:-1]
            max_val = np.max(test_set)
            if max_val >= thresh:
                return max_val
            else:
                return np.nan

        def custom_rolling_mean(x):
            return np.sum(x[:-1])

        # get invalid-raise/drop mask:
        raise_series = dataseries.rolling(raise_window_td, min_periods=2, closed="both")

        numba_boost = True
        if numba_boost:
            raise_check_boosted = numba.jit(raise_check, nopython=True)
            raise_series = raise_series.apply(
                raise_check_boosted, args=(thresh,), raw=True, engine="numba"
            )
        else:
            raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True)

        if raise_series.isna().all():
            return self

        # "unflag" values of insufficient deviation to their predecessors
        if slope is not None:
            w_mask = (
                pd.Series(dataseries.index).diff().dt.total_seconds()
                / freq_dt.total_seconds()
            ) > weight
            slope_mask = np.abs(dataseries.diff()) < slope
            to_unflag = raise_series.notna() & w_mask.values & slope_mask
            raise_series[to_unflag] = np.nan

        # calculate and apply the weighted mean weights (pseudo-harmonization):
        weights = (
            pd.Series(dataseries.index).diff(periods=2).shift(-1).dt.total_seconds()
            / freq_dt.total_seconds()
            / 2
        )

        weights.iloc[0] = 0.5 + (
            pd.Timestamp(dataseries.index[1]) - pd.Timestamp(dataseries.index[0])
        ).total_seconds() / (freq_dt.total_seconds() * 2)

        weights.iloc[-1] = 0.5 + (
            pd.Timestamp(dataseries.index[-1]) - pd.Timestamp(dataseries.index[-2])
        ).total_seconds() / (freq_dt.total_seconds() * 2)

        weights[weights > 1.5] = 1.5
        weights.index = dataseries.index
        weighted_data = dataseries.mul(weights)

        # rolling weighted mean calculation
        weighted_rolling_mean = weighted_data.rolling(
            average_window, min_periods=2, closed="both"
        )
        weights_rolling_sum = weights.rolling(
            average_window, min_periods=2, closed="both"
        )
        if numba_boost:
            custom_rolling_mean_boosted = numba.jit(custom_rolling_mean, nopython=True)
            weighted_rolling_mean = weighted_rolling_mean.apply(
                custom_rolling_mean_boosted, raw=True, engine="numba"
            )
            weights_rolling_sum = weights_rolling_sum.apply(
                custom_rolling_mean_boosted, raw=True, engine="numba"
            )
        else:
            weighted_rolling_mean = weighted_rolling_mean.apply(
                custom_rolling_mean, raw=True
            )
            weights_rolling_sum = weights_rolling_sum.apply(
                custom_rolling_mean, raw=True, engine="numba"
            )

        weighted_rolling_mean = weighted_rolling_mean / weights_rolling_sum
        # check means against critical raise value:
        to_flag = dataseries >= weighted_rolling_mean + (raise_series / raise_factor)
        to_flag &= raise_series.notna()
        self._flags[to_flag[to_flag].index, field] = flag

        return self

    @flagging()
    def flagMAD(
        self: "SaQC",
        field: str,
        window: Optional[str, int] = None,
        z: float = 3.5,
        min_residuals: Optional[int] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The function represents an implementation of the modyfied Z-score outlier detection method.

        See references [1] for more details on the algorithm.

        Note, that the test needs the input data to be sampled regularly (fixed sampling rate).

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
        z: float, default 3.5
            The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
        min_periods
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
        center
            Weather or not to center the target value in the scoring window. If `False`, the
            target value is the last value in the window.
        flag : float, default BAD
            flag to set.

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        flags : saqc.Flags
            The quality flags of data
            Flags values may have changed, relatively to the flags input.

        References
        ----------
        [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        """

        self = self.flagZScore(
            field,
            window=window,
            thresh=z,
            min_residuals=min_residuals,
            model_func=np.median,
            norm_func=lambda x: median_abs_deviation(
                x, scale="normal", nan_policy="omit"
            ),
            center=center,
            min_periods=min_periods,
            flag=flag,
        )

        return self

    @flagging()
    def flagOffset(
        self: "SaQC",
        field: str,
        tolerance: float,
        window: Union[int, str],
        thresh: Optional[float] = None,
        thresh_relative: Optional[float] = None,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        A basic outlier test that works on regularly and irregularly sampled data.

        The test classifies values/value courses as outliers by detecting not only a rise
        in value, but also, by checking for a return to the initial value level.

        Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with
        associated timestamps :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if

        1. :math:`|x_{n-1} - x_{n + s}| >` `thresh`, for all :math:`s \\in [0,1,2,...,k]`

        2. if `thresh_relative` > 0, :math:`x_{n + s} > x_{n - 1}*(1+` `thresh_relative` :math:`)`

        3. if `thresh_relative` < 0, :math:`x_{n + s} < x_{n - 1}*(1+` `thresh_relative` :math:`)`

        4. :math:`|x_{n-1} - x_{n+k+1}| <` `tolerance`

        5. :math:`|t_{n-1} - t_{n+k+1}| <` `window`

        Note, that this definition of a "spike" not only includes one-value outliers, but
        also plateau-ish value courses.

        Parameters
        ----------
        field : str
            The field in data.
        tolerance : float
            Maximum difference allowed, between the value, directly preceding and the value, directly succeeding an offset,
            to trigger flagging of the values forming the offset.
            See condition (4).
        window : {str, int}, default '15min'
            Maximum length allowed for offset value courses, to trigger flagging of the values forming the offset.
            See condition (5). Integer defined window length are only allowed for regularly sampled timeseries.
        thresh : float: {float, None}, default None
            Minimum difference between a value and its successors, to consider the successors an anomalous offset group.
            See condition (1). If None is passed, condition (1) is not tested.
        thresh_relative : {float, None}, default None
            Minimum relative change between a value and its successors, to consider the successors an anomalous offset group.
            See condition (2). If None is passed, condition (2) is not tested.
        flag : float, default BAD
            flag to set.

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        flags : saqc.Flags
            The quality flags of data
            Flags values may have changed, relatively to the flags input.

        Examples
        --------
        Below picture gives an abstract interpretation of the parameter interplay in case of a positive value jump,
        initialising an offset course.

        .. figure:: /resources/images/flagOffsetPic.png

           The four values marked red, are flagged, because (1) the initial value jump *exceeds* the value given by `thresh`,
           (2) the temporal extension of the group does *not exceed* the range given by `window` and (3) the returning
           value after the group, lies *within* the value range determined by `tolerance`


        .. plot::
           :context:
           :include-source: False

           import matplotlib
           import saqc
           import pandas as pd
           data = pd.DataFrame({'data':np.array([5,5,8,16,17,7,4,4,4,1,1,4])}, index=pd.date_range('2000',freq='1H', periods=12))


        Lets generate a simple, regularly sampled timeseries with an hourly sampling rate and generate an
        :py:class:`saqc.SaQC` instance from it.

        .. doctest:: flagOffsetExample

           >>> data = pd.DataFrame({'data':np.array([5,5,8,16,17,7,4,4,4,1,1,4])}, index=pd.date_range('2000',freq='1H', periods=12))
           >>> data
                                data
           2000-01-01 00:00:00     5
           2000-01-01 01:00:00     5
           2000-01-01 02:00:00     8
           2000-01-01 03:00:00    16
           2000-01-01 04:00:00    17
           2000-01-01 05:00:00     7
           2000-01-01 06:00:00     4
           2000-01-01 07:00:00     4
           2000-01-01 08:00:00     4
           2000-01-01 09:00:00     1
           2000-01-01 10:00:00     1
           2000-01-01 11:00:00     4
           >>> qc = saqc.SaQC(data)

        Now we are applying :py:meth:`~saqc.SaQC.flagOffset` and try to flag offset courses, that dont extend longer than
        *6 hours* in time (``window``) and that have an initial value jump higher than *2* (``thresh``), and that do return
        to the initial value level within a tolerance of *1.5* (``tolerance``).

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, tolerance=1.5, window='6H')
           >>> qc.plot('data') # doctest:+SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, tolerance=1.5, window='6H')
           >>> qc.plot('data')

        Note, that both, negative and positive jumps are considered starting points of negative or positive offsets.
        If you want to impose the additional condition, that the initial value jump must exceed *+90%* of the value level,
        you can additionally set the ``thresh_relative`` parameter:

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=.9, tolerance=1.5, window='6H')
           >>> qc.plot('data') # doctest:+SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=.9, tolerance=1.5, window='6H')
           >>> qc.plot('data')

        Now, only positive jumps, that exceed a value gain of *+90%* are considered starting points of offsets.

        In the same way, you can aim for only negative offsets, by setting a negative relative threshold. The below
        example only flags offsets, that fall off by at least *50 %* in value, with an absolute value drop of at least *2*.

        .. doctest:: flagOffsetExample

           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=-.5, tolerance=1.5, window='6H')
           >>> qc.plot('data') # doctest:+SKIP

        .. plot::
           :context: close-figs
           :include-source: False

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagOffset("data", thresh=2, thresh_relative=-.5, tolerance=1.5, window='6H')
           >>> qc.plot('data')


        References
        ----------
        The implementation is a time-window based version of an outlier test from the UFZ Python library,
        that can be found here:

        https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py

        """
        if (thresh is None) and (thresh_relative is None):
            raise ValueError(
                "At least one of parameters 'thresh' and 'thresh_relative' has to be given. Got 'thresh'=None, "
                "'thresh_relative'=None instead."
            )
        if thresh is None:
            thresh = 0

        dat = self._data[field].dropna()
        if thresh_relative is not None:
            rel_jumps = np.sign(thresh_relative) * dat > np.sign(
                thresh_relative
            ) * dat.shift(+1) * (1 + thresh_relative)

        data_diff = dat.diff()
        initial_jumps = data_diff.abs() > thresh
        if thresh_relative:
            initial_jumps &= rel_jumps
        return_in_time = (
            dat[::-1]
            .rolling(window, min_periods=2)
            .apply(lambda x: np.abs(x[-1] - x[:-1]).min() < tolerance, raw=True)[::-1]
            .astype(bool)
        )
        return_in_time = return_in_time & initial_jumps.reindex(
            dat.index, fill_value=False
        ).shift(-1, fill_value=False)
        offset_start_candidates = dat[return_in_time]
        win_delta = pd.Timedelta(window)
        corners = pd.Series(False, index=dat.index)
        to_flag = pd.Series(False, index=dat.index)
        ns = pd.Timedelta("1ns")
        for c in zip(offset_start_candidates.index, offset_start_candidates.values):
            ret = (dat[c[0]] - dat[c[0] + ns : c[0] + win_delta]).abs()[1:] < tolerance
            if not ret.empty:
                r = ret.idxmax()
                chunk = dat[c[0] : r]
                sgn = np.sign(chunk[1] - c[1])
                t_val = ((chunk[1:-1] - c[1]) * sgn > thresh).all()
                r_val = True
                if thresh_relative:
                    r_val = (
                        np.sign(thresh_relative) * chunk[1:-1]
                        > np.sign(thresh_relative) * c[1] * (1 + thresh_relative)
                    ).all()
                if t_val and r_val and (not corners[c[0]]):
                    flag_i = dat[c[0] + ns : chunk.index[-1] - ns].index
                    to_flag[flag_i] = True
                    corners.loc[flag_i[-1]] = True
        to_flag = to_flag.reindex(self._data[field].index, fill_value=False)

        self._flags[to_flag, field] = flag
        return self

    @flagging()
    def flagByGrubbs(
        self: "SaQC",
        field: str,
        window: Union[str, int],
        alpha: float = 0.05,
        min_periods: int = 8,
        pedantic: bool = False,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        The function flags values that are regarded outliers due to the grubbs test.

        See reference [1] for more information on the grubbs tests definition.

        The (two-sided) test gets applied onto data chunks of size "window". The tests
        application  will be iterated on each data-chunk under test, till no more
        outliers are detected in that chunk.

        Note, that the test performs poorely for small data chunks (resulting in heavy
        overflagging). Therefor you should select "window" so that every window contains
        at least > 8 values and also adjust the min_periods values accordingly.

        Note, that the data to be tested by the grubbs test are expected to be distributed
        "normalish".

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.

        window : {int, str}
            The size of the window you want to use for outlier testing. If an integer is
            passed, the size refers to the number of periods of every testing window. If a
            string is passed, it has to be an offset string, and will denote the total
            temporal extension of every window.

        alpha : float, default 0.05
            The level of significance, the grubbs test is to be performed at. (between 0 and 1)

        min_periods : int, default 8
            The minimum number of values that have to be present in an interval under test,
            for a grubbs test result to be accepted. Only makes sence in case `window` is
            an offset string.

        pedantic: boolean, default False
            If True, every value gets checked twice for being an outlier. Ones in the
            initial rolling window and one more time in a rolling window that is lagged
            by half the windows delimeter (window/2). Recommended for avoiding false
            positives at the window edges. Only available when rolling with integer
            defined window size.

        flag : float, default BAD
            flag to set.

        Returns
        -------
        saqc.SaQC

        References
        ----------
        introduction to the grubbs test:

        [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
        """
        datcol = self._data[field].copy()
        rate = getFreqDelta(datcol.index)

        # if timeseries that is analyzed, is regular,
        # window size can be transformed to a number of periods:
        if rate and isinstance(window, str):
            window = pd.Timedelta(window) // rate

        to_group = pd.DataFrame(data={"ts": datcol.index, "data": datcol})
        to_flag = pd.Series(False, index=datcol.index)

        # period number defined test intervals
        if isinstance(window, int):
            grouper_series = pd.Series(
                data=np.arange(0, datcol.shape[0]), index=datcol.index
            )
            grouper_series_lagged = grouper_series + (window / 2)
            grouper_series = grouper_series.transform(lambda x: x // window)
            grouper_series_lagged = grouper_series_lagged.transform(
                lambda x: x // window
            )
            partitions = to_group.groupby(grouper_series)
            partitions_lagged = to_group.groupby(grouper_series_lagged)

        # offset defined test intervals:
        else:
            partitions = to_group.groupby(pd.Grouper(freq=window))
            partitions_lagged = []

        for _, partition in partitions:
            if partition.shape[0] > min_periods:
                detected = smirnov_grubbs.two_sided_test_indices(
                    partition["data"].values, alpha=alpha
                )
                detected = partition["ts"].iloc[detected]
                to_flag[detected.index] = True

        if isinstance(window, int) and pedantic:
            to_flag_lagged = pd.Series(False, index=datcol.index)

            for _, partition in partitions_lagged:
                if partition.shape[0] > min_periods:
                    detected = smirnov_grubbs.two_sided_test_indices(
                        partition["data"].values, alpha=alpha
                    )
                    detected = partition["ts"].iloc[detected]
                    to_flag_lagged[detected.index] = True

            to_flag &= to_flag_lagged

        self._flags[to_flag, field] = flag
        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field"],
        multivariate=True,
        handles_target=False,
    )
    def flagCrossStatistics(
        self: "SaQC",
        field: Sequence[str],
        thresh: float,
        method: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Function checks for outliers relatively to the "horizontal" input data axis.

        For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
        for outlier detection:

        1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
           :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
        2. for every :math:`0 <= i <= K`, the value
           :math:`m_j = median(\\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\\})` is calculated
        3. for every :math:`0 <= i <= K`, the set
           :math:`\\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\\}` is tested for outliers with the
           specified method (`cross_stat` parameter).

        Parameters
        ----------
        field : list of str
            List of fieldnames in data, determining wich variables are to be included into the flagging process.

        thresh : float
            Threshold which the outlier score of an value must exceed, for being flagged an outlier.

        method : {'modZscore', 'Zscore'}, default 'modZscore'
            Method used for calculating the outlier scores.

            * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
            * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
              See References [1]

        flag : float, default BAD
            flag to set.

        Returns
        -------
        saqc.SaQC


        Notes
        -----

        The input variables dont necessarily have to be aligned. If the variables are unaligned, scoring
        and flagging will be only performed on the subset of inices shared among all input variables.


        References
        ----------
        [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        """

        fields = toSequence(field)

        df = self._data[fields].loc[self._data[fields].index_of("shared")].to_df()

        if isinstance(method, str):

            if method == "modZscore":
                MAD_series = df.subtract(df.median(axis=1), axis=0).abs().median(axis=1)
                diff_scores = (
                    (0.6745 * (df.subtract(df.median(axis=1), axis=0)))
                    .divide(MAD_series, axis=0)
                    .abs()
                )

            elif method == "Zscore":
                diff_scores = (
                    df.subtract(df.mean(axis=1), axis=0)
                    .divide(df.std(axis=1), axis=0)
                    .abs()
                )

            else:
                raise ValueError(method)

        else:

            try:
                stat = getattr(df, method.__name__)(axis=1)
            except AttributeError:
                stat = df.aggregate(method, axis=1)

            diff_scores = df.subtract(stat, axis=0).abs()

        mask = diff_scores > thresh
        if mask.empty:
            return self

        for f in fields:
            m = mask[f].reindex(index=self._flags[f].index, fill_value=False)
            self._flags[m, f] = flag

        return self

    @flagging()
    def flagZScore(
        self: "SaQC",
        field: str,
        window: str | int | None = None,
        thresh: float = 3,
        min_residuals: int | None = None,
        min_periods: int | None = None,
        model_func: Callable = np.nanmean,
        norm_func: Callable = np.nanstd,
        center: bool = True,
        flag: float = BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Flag data where its (rolling) Zscore exceeds a threshold.

        The function implements flagging derived from a basic Zscore calculation.
        To handle non stationary data, the Zscoring can be applied with a rolling window.
        Therefor, the function allows for a minimum residual to be specified in order to mitigate overflagging in
        local regimes of low variance.

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
        thresh
            Cutoff level for the Zscores, above which associated points are getting flagged.
        min_residuals
            Minimum residual level points must have to be considered outliers.
        min_periods
            Minimum number of valid meassurements in a scoring window, to consider the resulting score valid.
        model_func
            Function to calculate the center moment in every window.
        norm_func
            Function to calculate the scaling for every window
        center
            Weather or not to center the target value in the scoring window. If `False`, the
            target value is the last value in the window.

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
        containing the value :math:`y_{K}` which is to be checked.
        (The index of :math:`K` depends on the selection of the parameter `center`.)

        2. The "moment" :math:`M` for the window gets calculated via :math:`M=` `model_func(:math:`W`)

        3. The "scaling" :math:`N` for the window gets calculated via :math:`N=` `norm_func(:math:`W`)

        4. The "score" :math:`S` for the point :math:`x_{k}`gets calculated via :math:`S=(x_{k} - M) / N`

        5. Finally, :math:`x_{k}` gets flagged, if :math:`|S| >` `thresh` and :math:`|M - x_{k}| >= `min_residuals`
        """
        datser = self._data[field]
        if min_residuals is None:
            min_residuals = 0

        score, model, _ = _univarScoring(
            datser,
            window=window,
            norm_func=norm_func,
            model_func=model_func,
            center=center,
            min_periods=min_periods,
        )
        to_flag = (score.abs() > thresh) & ((model - datser).abs() >= min_residuals)
        self._flags[to_flag, field] = flag
        return self


def _evalStrayLabels(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: Sequence[str],
    reduction_range: Optional[str] = None,
    reduction_drop_flagged: bool = False,  # TODO: still a case ?
    reduction_thresh: float = 3.5,
    reduction_min_periods: int = 1,
    at_least_one: bool = True,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function "reduces" an observations flag to components of it, by applying MAD
    (See references) test onto every components temporal surrounding.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the labels to be evaluated.

    flags : saqc.Flags
        Container to store quality flags to data.

    target : list of str
        A list of strings, holding the column names of the variables, the stray labels
        shall be projected onto.

    val_frame : (N,M) pd.DataFrame
        Input NxM DataFrame of observations, where N is the number of observations and
        M the number of components per observation.

    to_flag_frame : pandas.DataFrame
        Input dataframe of observations to be tested, where N is the number of
        observations and M the number of components per observation.

    reduction_range : {None, str}
        An offset string, denoting the range of the temporal surrounding to include
        into the MAD testing. If ``None`` is passed, no testing will be performed and
        all targets will have the stray flag projected.

    reduction_drop_flagged : bool, default False
        Wheather or not to drop flagged values other than the value under test, from the
        temporal surrounding before checking the value with MAD.

    reduction_thresh : float, default 3.5
        The `critical` value, controlling wheather the MAD score is considered
        referring to an outlier or not. Higher values result in less rigid flagging.
        The default value is widely used in the literature. See references section
        for more details ([1]).

    at_least_one : bool, default True
        If none of the variables, the outlier label shall be reduced to, is an outlier
        with regard to the test, all (True) or none (False) of the variables are flagged

    flag : float, default BAD
        flag to set.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    val_frame = data[target].to_df()
    stray_detects = flags[field] > UNFLAGGED
    stray_detects = stray_detects[stray_detects]
    to_flag_frame = pd.DataFrame(False, columns=target, index=stray_detects.index)

    if reduction_range is None:
        for field in to_flag_frame.columns:
            flags[to_flag_frame.index, field] = flag
        return data, flags

    for var in target:
        for index in enumerate(to_flag_frame.index):

            index_slice = slice(
                index[1] - pd.Timedelta(reduction_range),
                index[1] + pd.Timedelta(reduction_range),
            )
            test_slice = val_frame[var][index_slice].dropna()

            # check, wheather value under test is sufficiently centered:
            first = test_slice.first_valid_index()
            last = test_slice.last_valid_index()
            min_range = pd.Timedelta(reduction_range) / 4

            if (
                pd.Timedelta(index[1] - first) < min_range
                or pd.Timedelta(last - index[1]) < min_range
            ):
                polydeg = 0
            else:
                polydeg = 2

            if reduction_drop_flagged:
                test_slice = test_slice.drop(to_flag_frame.index, errors="ignore")

            if test_slice.shape[0] < reduction_min_periods:
                to_flag_frame.loc[index[1], var] = True
                continue

            x = test_slice.index.values.astype(float)
            x_0 = x[0]
            x = (x - x_0) / 10**12

            polyfitted = poly.polyfit(y=test_slice.values, x=x, deg=polydeg)

            testval = poly.polyval(
                (float(index[1].to_numpy()) - x_0) / 10**12, polyfitted
            )
            testval = val_frame[var][index[1]] - testval

            resids = test_slice.values - poly.polyval(x, polyfitted)
            med_resids = np.median(resids)
            MAD = np.median(np.abs(resids - med_resids))
            crit_val = 0.6745 * (abs(med_resids - testval)) / MAD

            if crit_val > reduction_thresh:
                to_flag_frame.loc[index[1], var] = True

    if at_least_one:
        to_flag_frame[~to_flag_frame.any(axis=1)] = True

    for field in to_flag_frame.columns:
        col = to_flag_frame[field]
        flags[col[col].index, field] = flag

    return data, flags
