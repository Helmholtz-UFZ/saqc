#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Callable, Tuple, Union
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.core import register, Flags


@register(mask=["field"], demask=[], squeeze=[])
def transform(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    func: Callable[[pd.Series], pd.Series],
    freq: Optional[Union[float, str]] = None,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function to transform data columns with a transformation that maps series onto series of the same length.

    Note, that flags get preserved.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-transformed.
    flags : saqc.Flags
        Container to store quality flags to data.
    func : Callable[{pd.Series, np.array}, np.array]
        Function to transform data[field] with.
    freq : {None, float, str}, default None
        Determines the segmentation of the data into partitions, the transformation is applied on individually

        * ``np.inf``: Apply transformation on whole data set at once
        * ``x`` > 0 : Apply transformation on successive data chunks of periods length ``x``
        * Offset String : Apply transformation on successive partitions of temporal extension matching the passed offset
          string

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    val_ser = data[field].copy()
    # partitioning
    if not freq:
        freq = val_ser.shape[0]

    if isinstance(freq, str):
        grouper = pd.Grouper(freq=freq)
    else:
        grouper = pd.Series(data=np.arange(0, val_ser.shape[0]), index=val_ser.index)
        grouper = grouper.transform(lambda x: int(np.floor(x / freq)))

    partitions = val_ser.groupby(grouper)

    for _, partition in partitions:
        if partition.empty:
            continue
        val_ser[partition.index] = func(partition)

    data[field] = val_ser
    return data, flags
