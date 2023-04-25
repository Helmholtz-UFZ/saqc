#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

from saqc.core import register

if TYPE_CHECKING:
    from saqc import SaQC


class TransformationMixin:
    @register(mask=["field"], demask=[], squeeze=[])
    def transform(
        self: "SaQC",
        field: str,
        func: Callable[[pd.Series | np.ndarray], pd.Series],
        freq: float | str | None = None,
        **kwargs,
    ) -> "SaQC":
        """
        Transform data by applying a custom function on data chunks of variable size. Existing flags are preserved.

        Parameters
        ----------
        func :
            Transformation function.

        freq :
            Size of the data window. The transformation is applied on each window individually

            * ``None``: Apply transformation on the entire data set at once
            * ``int`` : Apply transformation on successive data chunks of the given length. Must be grater than 0.
            * Offset String : Apply transformation on successive data chunks of the given temporal extension.
        """
        val_ser = self._data[field].copy()
        # partitioning
        if not freq:
            freq = len(val_ser)

        if isinstance(freq, str):
            grouper = pd.Grouper(freq=freq)
        else:
            grouper = pd.Series(data=np.arange(0, len(val_ser)), index=val_ser.index)
            grouper = grouper.transform(lambda x: int(np.floor(x / freq)))

        partitions = val_ser.groupby(grouper)

        for _, partition in partitions:
            if partition.empty:
                continue
            val_ser[partition.index] = func(partition)

        self._data[field] = val_ser
        return self
