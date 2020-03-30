#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist



def _isValid(data, max_nan_total, max_nan_consec):
    if (max_nan_total is np.inf) & (max_nan_consec is np.inf):
        return True

    nan_mask = data.isna()

    if nan_mask.sum() <= max_nan_total:
        if max_nan_consec is np.inf:
            return True
        elif nan_mask.rolling(window=max_nan_consec + 1).sum().max() <= max_nan_consec:
            return True
        else:
            return False
    else:
        return False

# ts_transformations
def identity(ts):
    return ts


def difference(ts):
    return pd.Series.diff(ts)


def derivative(ts, unit='1min'):
    return ts/(deltaT(ts, unit=unit))


def deltaT(ts, unit='1min'):
    return ts.index.to_series().diff().dt.total_seconds() / pd.Timedelta(unit).total_seconds()


def rateOfChange(ts):
    return ts.diff/ts


def relativeDifference(ts):
    return ts - 0.5*(ts.shift(+1) + ts.shift(-1))


def scale(ts, target_range=1, projection_point=None):
    if not projection_point:
        projection_point = ts.abs().max()
    return (ts / projection_point) * target_range


def normScale(ts):
    ts_min = ts.min()
    return (ts - ts_min) / (ts.max() - ts_min)


def nBallClustering(X, ball_radius=None):
    x_len = X.shape[0]
    x_cols = X.shape[1]

    if not ball_radius:
        ball_radius = 0.1 / np.log(x_len)**(1/x_cols)
    exemplars = [X[0, :]]
    members = [[]]
    iterator = np.nditer(X, flags=['f_index'], order='C')
    index = 0
    for point in X:
        dists = np.linalg.norm(point - np.array(exemplars), axis=1)
        min_index = dists.argmin()
        if dists[min_index] < ball_radius:
            members[min_index].append(index)
        else:
            exemplars.append(X[index])
            members.append([index])
        index += 1

    return exemplars, members


def stdQC(data, max_nan_total=np.inf, max_nan_consec=np.inf):
    """Pandas built in function for statistical moments have quite poor nan- control, so here comes a wrapper that
    will return the standart deviation for a given series input, if the total number of nans in the series does
    not exceed "max_nan_total" and the number of consecutive nans does not exceed max_nan_consec.

    :param data             Pandas Series. The data series, the standart deviation shall be calculated of.
    :param max_nan_total    Integer. Number of np.nan entries allowed to be contained in the series
    :param max_nan_consec   Integer. Maximal number of consecutive nan entries allowed to occure in data.
    """
    if _isValid(data, max_nan_total, max_nan_consec):
        return data.std()
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
        return data.var()
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
        return data.mean()
    return np.nan