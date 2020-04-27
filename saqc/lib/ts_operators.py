#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numba as nb
import math
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr
import logging
logger = logging.getLogger("SaQC")

# CONSISTENCY-NOTE:
# ALL transformations can handle np.array and pd.Series as input (excluded the transformations needing timestamp
# informations for calculation). Although some transformations retain pd.Series index information -
# some others do not. Use dataseries' .transform / .resample / ... methods to apply transformations to
# dataseries/dataframe columns, so you can be sure to keep index informations.

def identity(ts):
    return ts


def zeroLog(ts):
    log_ts = np.log(ts)
    log_ts[log_ts == -np.inf] = np.nan
    return log_ts


def difference(ts):
    # NOTE: index of input series gets lost!
    return np.diff(ts, prepend=np.nan)


def derivative(ts, unit="1min"):
    return ts / (deltaT(ts, unit=unit))


def deltaT(ts, unit="1min"):
    return ts.index.to_series().diff().dt.total_seconds() / pd.Timedelta(unit).total_seconds()


def rateOfChange(ts):
    return difference(ts) / ts


def relativeDifference(ts):
    res = ts - 0.5 * (np.roll(ts, +1) + np.roll(ts, -1))
    res[0] = np.nan
    res[-1] = np.nan
    return res


def scale(ts, target_range=1, projection_point=None):
    if not projection_point:
        projection_point = np.max(np.abs(ts))
    return (ts / projection_point) * target_range


def normScale(ts):
    ts_min = ts.min()
    return (ts - ts_min) / (ts.max() - ts_min)


def standardizeByMean(ts):
    return (ts - np.mean(ts))/np.std(ts, ddof=1)


def standardizeByMedian(ts):
    return (ts - np.median(ts))/iqr(ts, nan_policy='omit')


def _kNN(in_arr, n_neighbors, algorithm="ball_tree"):
    # in: array only
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(in_arr.reshape(-1, 1))
    return nbrs.kneighbors()


def kNNMaxGap(in_arr, n_neighbors, algorithm='ball_tree'):
    in_arr = np.asarray(in_arr)
    dist, *_ = _kNN(in_arr, n_neighbors, algorithm=algorithm)
    sample_size = dist.shape[0]
    to_gap = np.append(np.array([[0] * sample_size]).T, dist, axis=1)
    max_gap_ind = np.diff(to_gap, axis=1).argmax(axis=1)
    return dist[range(0, sample_size), max_gap_ind]


def kNNSum(in_arr, n_neighbors, algorithm="ball_tree"):
    in_arr = np.asarray(in_arr)
    dist, *_ = _kNN(in_arr, n_neighbors, algorithm=algorithm)
    return dist.sum(axis=1)


@nb.njit
def _max_consecutive_nan(arr):
    max_ = 0
    current = 0
    idx = 0
    while idx < arr.size:
        while idx < arr.size and math.isnan(arr[idx]):
            current += 1
            idx += 1
        if current > max_:
            max_ = current
        current = 0
        idx += 1
    return max_


def _isValid(data, max_nan_total, max_nan_consec):
    if (max_nan_total is np.inf) & (max_nan_consec is np.inf):
        return True

    nan_mask = np.isnan(data)

    if nan_mask.sum() <= max_nan_total:
        if max_nan_consec is np.inf:
            return True
        elif _max_consecutive_nan(np.asarray(data)) <= max_nan_consec:
            return True
        else:
            return False
    else:
        return False


def stdQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the standart deviation for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the standart deviation shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _isValid(data, max_nan_total, max_nan_consec):
        return np.std(data, ddof=1)
    return np.nan


def varQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the variance for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the variance shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _isValid(data, max_nan_total, max_nan_consec):
        return np.var(data, ddof=1)
    return np.nan


def meanQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the mean for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the mean shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _isValid(data, max_nan_total, max_nan_consec):
        return np.mean(data)
    return np.nan


def interpolateNANs(data, method, order=2, inter_limit=2, downcast_interpolation=False, return_chunk_bounds=False):
    """
    The function interpolates nan-values (and nan-grids) in timeseries data. It can be passed all the method keywords
    from the pd.Series.interpolate method and will than apply this very methods. Note, that the inter_limit keyword
    really restricts the interpolation to chunks, not containing more than "inter_limit" nan entries
    (thereby opposing the limit keyword of pd.Series.interpolate).

    :param data:                    pd.Series. The data series to be interpolated
    :param method:                  String. Method keyword designating interpolation method to use.
    :param order:                   Integer. If your desired interpolation method needs an order to be passed -
                                    here you pass it.
    :param inter_limit:             Integer. Default = 2. Limit up to whitch nan - gaps in the data get interpolated.
                                    Its default value suits an interpolation that only will apply on an inserted
                                    frequency grid. (regularization by interpolation)
    :param downcast_interpolation:  Boolean. Default False. If True:
                                    If a data chunk not contains enough values for interpolation of the order "order",
                                    the highest order possible will be selected for that chunks interpolation."
    :param return_chunk_bounds:     Boolean. Default False. If True:
                                    Additionally to the interpolated data, the start and ending points of data chunks
                                    not containing no series consisting of more then "inter_limit" nan values,
                                    are calculated and returned.
                                    (This option fits requirements of the "_interpolate" functions use in the context of
                                    saqc harmonization mainly.)

    :return:
    """

    data = pd.Series(data)
    gap_mask = (data.rolling(inter_limit, min_periods=0).apply(lambda x: np.sum(np.isnan(x)), raw=True)) != inter_limit

    if inter_limit == 2:
        gap_mask = gap_mask & gap_mask.shift(-1, fill_value=True)
    else:
        gap_mask = (
            gap_mask.replace(True, np.nan).fillna(method="bfill", limit=inter_limit).replace(np.nan, True).astype(bool)
        )

    if return_chunk_bounds:
        # start end ending points of interpolation chunks have to be memorized to block their flagging:
        chunk_switches = gap_mask.astype(int).diff()
        chunk_starts = chunk_switches[chunk_switches == -1].index
        chunk_ends = chunk_switches[(chunk_switches.shift(-1) == 1)].index
        chunk_bounds = chunk_starts.join(chunk_ends, how="outer", sort=True)

    data = data[gap_mask]

    if method in ["linear", "time"]:

        data.interpolate(method=method, inplace=True, limit=1, limit_area="inside")

    else:
        dat_name = data.name
        gap_mask = (~gap_mask).cumsum()
        data = pd.merge(gap_mask, data, how="inner", left_index=True, right_index=True)

        def _interpolWrapper(x, wrap_order=order, wrap_method=method):
            if x.count() > wrap_order:
                try:
                    return x.interpolate(method=wrap_method, order=int(wrap_order))
                except (NotImplementedError, ValueError):
                    logger.warning(
                        "Interpolation with method {} is not supported at order {}. "
                        "Interpolation will be performed with order {}".format(
                            method, str(wrap_order), str(wrap_order - 1)
                        )
                    )
                    return _interpolWrapper(x, int(wrap_order - 1), wrap_method)
            elif x.size < 3:
                return x
            else:
                if downcast_interpolation:
                    return _interpolWrapper(x, int(x.count() - 1), wrap_method)
                else:
                    return x

        data = data.groupby(data.columns[0]).transform(_interpolWrapper)
        # squeezing the 1-dimensional frame resulting from groupby for consistency reasons
        data = data.squeeze(axis=1)
        data.name = dat_name
    if return_chunk_bounds:
        return data, chunk_bounds
    else:
        return data


def leaderClustering(in_arr, ball_radius=None):
    x_len = in_arr.shape[0]
    x_cols = in_arr.shape[1]

    if not ball_radius:
        ball_radius = 0.1 / np.log(x_len) ** (1 / x_cols)
    exemplars = [in_arr[0, :]]
    members = [[]]
    for index, point in in_arr:
        dists = np.linalg.norm(point - np.array(exemplars), axis=1)
        min_index = dists.argmin()
        if dists[min_index] < ball_radius:
            members[min_index].append(index)
        else:
            exemplars.append(in_arr[index])
            members.append([index])
    ex_indices = [x[0] for x in members]
    return exemplars, members, ex_indices