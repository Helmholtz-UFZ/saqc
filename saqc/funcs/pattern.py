#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING

import dtw
import numpy as np
import pandas as pd

from saqc import BAD
from saqc.core import flagging
from saqc.lib.rolling import removeRollingRamps

if TYPE_CHECKING:
    from saqc import SaQC


def calculateDistanceByDTW(
    data: pd.Series, reference: pd.Series, forward=True, normalize=True
):
    """
    Calculate the DTW-distance of data to pattern in a rolling calculation.

    The data is compared to pattern in a rolling window.
    The size of the rolling window is determined by the timespan defined
    by the first and last timestamp of the reference data's datetime index.

    For details see the linked functions in the `See Also` section.

    Parameters
    ----------
    data : pd.Series
        Data series. Must have datetime-like index, and must be regularly sampled.

    reference : : pd.Series
        Reference series. Must have datetime-like index, must not contain NaNs
        and must not be empty.

    forward: bool, default True
        If `True`, the distance value is set on the left edge of the data chunk. This
        means, with a perfect match, `0.0` marks the beginning of the pattern in
        the data. If `False`, `0.0` would mark the end of the pattern.

    normalize : bool, default True
        If `False`, return unmodified distances.
        If `True`, normalize distances by the number of observations in the reference.
        This helps to make it easier to find a good cutoff threshold for further
        processing. The distances then refer to the mean distance per datapoint,
        expressed in the datas units.

    Returns
    -------
    distance : pd.Series

    Notes
    -----
    The data must be regularly sampled, otherwise a ValueError is raised.
    NaNs in the data will be dropped before dtw distance calculation.

    See Also
    --------
    flagPatternByDTW : flag data by DTW
    """
    if reference.hasnans or reference.empty:
        raise ValueError("reference must not have nan's and must not be empty.")

    winsz: pd.Timedelta = reference.index.max() - reference.index.min()
    reference = reference.to_numpy()

    def isPattern(chunk):
        if forward:
            return dtw.accelerated_dtw(chunk[::-1], reference, "euclidean")[0]
        else:
            return dtw.accelerated_dtw(chunk, reference, "euclidean")[0]

    # generate distances, excluding NaNs
    nonas = data.dropna()
    rollover = nonas[::-1] if forward else nonas
    arr = rollover.rolling(winsz, closed="both").apply(isPattern, raw=True).to_numpy()
    distances = pd.Series(arr[::-1] if forward else arr, index=nonas.index)
    removeRollingRamps(distances, window=winsz, inplace=True)

    if normalize:
        distances /= len(reference)

    return distances.reindex(index=data.index)  # reinsert NaNs


class PatternMixin:
    # todo should we mask `reference` even if the func fail if reference has NaNs
    @flagging()
    def flagPatternByDTW(
        self: "SaQC",
        field,
        reference,
        max_distance=0.0,
        normalize=True,
        plot=False,
        flag=BAD,
        **kwargs,
    ) -> "SaQC":
        """
        Pattern Recognition via Dynamic Time Warping.

        The steps are:
        1. work on a moving window

        2. for each data chunk extracted from each window, a distance to the given pattern
           is calculated, by the dynamic time warping algorithm [1]

        3. if the distance is below the threshold, all the data in the window gets flagged

        Parameters
        ----------
        reference :
            The name in `data` which holds the pattern. The pattern must not have NaNs,
            have a datetime index and must not be empty.

        max_distance :
            Maximum dtw-distance between chunk and pattern, if the distance is lower than
            ``max_distance`` the data gets flagged. With default, ``0.0``, only exact
            matches are flagged.

        normalize :
            If `False`, return unmodified distances.
            If `True`, normalize distances by the number of observations of the reference.
            This helps to make it easier to find a good cutoff threshold for further
            processing. The distances then refer to the mean distance per datapoint,
            expressed in the datas units.

        plot :
            Show a calibration plot, which can be quite helpful to find the right threshold
            for `max_distance`. It works best with `normalize=True`. Do not use in automatic
            setups / pipelines. The plot show three lines:

            - data: the data the function was called on
            - distances: the calculated distances by the algorithm
            - indicator: have to distinct levels: `0` and the value of `max_distance`.
              If `max_distance` is `0.0` it defaults to `1`. Everywhere where the
              indicator is not `0` the data will be flagged.

        Notes
        -----
        The window size of the moving window is set to equal the temporal extension of the
        reference datas datetime index.

        References
        ----------
        Find a nice description of underlying the Dynamic Time Warping Algorithm here:

        [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
        """
        ref = self._data[reference]
        dat = self._data[field]

        distances = calculateDistanceByDTW(dat, ref, forward=True, normalize=normalize)
        winsz = ref.index.max() - ref.index.min()

        # prevent nan propagation
        distances = distances.fillna(max_distance + 1)

        # find minima filter by threshold
        fw_min = distances[::-1].rolling(window=winsz, closed="both").min()[::-1]
        bw_min = distances.rolling(window=winsz, closed="both").min()
        minima = (fw_min == bw_min) & (distances <= max_distance)

        # Propagate True's to size of pattern.
        mask = minima.rolling(window=winsz, closed="both").sum() > 0

        if plot:
            df = pd.DataFrame()
            df["data"] = dat
            df["distances"] = distances
            df["indicator"] = mask.astype(float) * (max_distance or 1)
            df.plot()

        self._flags[mask, field] = flag
        return self
