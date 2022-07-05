#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

from saqc.core.register import register

if TYPE_CHECKING:
    from saqc.core.core import SaQC


class TransformationMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def transform(
        self: "SaQC",
        field: str,
        func: Callable[[pd.Series], pd.Series],
        freq: Optional[Union[float, str]] = None,
        **kwargs,
    ) -> "SaQC":
        """
        Function to transform data columns with a transformation that maps series onto series of the same length.

        Note, that flags get preserved.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-transformed.

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
        saqc.SaQC
        """
        val_ser = self._data[field].copy()
        # partitioning
        if not freq:
            freq = val_ser.shape[0]

        if isinstance(freq, str):
            grouper = pd.Grouper(freq=freq)
        else:
            grouper = pd.Series(
                data=np.arange(0, val_ser.shape[0]), index=val_ser.index
            )
            grouper = grouper.transform(lambda x: int(np.floor(x / freq)))

        partitions = val_ser.groupby(grouper)

        for _, partition in partitions:
            if partition.empty:
                continue
            val_ser[partition.index] = func(partition)

        self._data[field] = val_ser
        return self
