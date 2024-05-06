#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd

import saqc
from saqc.core import DictOfSeries, Flags, register
from saqc.lib.checking import validateFuncSelection, validateMinPeriods, validateWindow
from saqc.lib.tools import getFreqDelta

if TYPE_CHECKING:
    from saqc import SaQC


class RollingMixin:
    @register(
        mask=["field"], demask=[], squeeze=[], multivariate=True, handles_target=True
    )
    def rolling(
        self: "SaQC",
        field: str | list[str],
        window: str | int,
        target: str | list[str] = None,
        func: Callable[[pd.Series], np.ndarray] | str = "mean",
        min_periods: int = 0,
        center: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Calculate a rolling-window function on the data.

        Note, that the new data gets assigned the worst flag present in the window it was aggregated from.

        Note, That you also can select multiple fields to get a rolling calculation over those.

        Parameters
        ----------
        window :
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.

        func : default mean
            Function to roll with.

        min_periods :
            The minimum number of periods to get a valid value

        center :
            If True, center the rolling window.

        Notes
        -----
        .. figure:: /resources/images/horizontalAxisRollingExample.png

           Rolling over multiple variables.
        """
        # HINT: checking in  _roll
        if target and (len(target) > 1) and (len(field) != len(target)):
            raise ValueError(
                f"""If multiple targets are given, per-field application of rolling is conducted and the number of 
                fields has to equal the number of targets.\n Got: \n Fields={field} \n Targets={target}"""
            )

        if target and (len(field) > 1) and (len(target) == 1):
            target = target[0]
            if target not in self._data.columns:
                self[target] = saqc.SaQC(
                    pd.Series(
                        np.nan, index=self.data[field].to_pandas().index, name=target
                    )
                )

            self._data, self._flags = _hroll(
                data=self._data,
                field=field,
                flags=self._flags,
                target=target,
                window=window,
                func=func,
                min_periods=min_periods,
                center=center,
            )

        else:
            if target:
                for ft in zip(field, target):
                    self = self.copyField(ft[0], target=ft[1], overwrite=True)
                field = target
            for f in field:
                self._data, self._flags = _roll(
                    data=self._data,
                    field=f,
                    flags=self._flags,
                    window=window,
                    func=func,
                    min_periods=min_periods,
                    center=center,
                    **kwargs,
                )
        return self


def _hroll(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    window: str | int,
    func: Callable[[pd.Series], np.ndarray] | str = "mean",
    min_periods: int = 0,
    center: bool = True,
    **kwargs,
):

    if isinstance(window, str):
        freq = getFreqDelta(data[field].to_pandas().index)
        if freq is None:
            raise ValueError(
                f"Rolling over more than one column is only supported if either the data has a unitary"
                f'sampling rate, or window is an integer. "{window}" was passed and combined {field} '
                f"index is not unitarily sampled"
            )
        else:
            window = int(np.floor(pd.Timedelta(window) / freq))

    views = np.lib.stride_tricks.sliding_window_view(
        data[field].to_pandas(), (window, len(field))
    )
    f_views = np.lib.stride_tricks.sliding_window_view(
        pd.DataFrame({f: flags[f] for f in field}), (window, len(field))
    )
    frame = pd.DataFrame(
        views.reshape(views.shape[0], views.shape[1] * views.shape[2] * views.shape[3])
    )
    if isinstance(func, str) and hasattr(pd.DataFrame, func):
        result = getattr(frame, func)(axis=1)
    else:
        result = frame.apply(func, axis=1)

    insuff_periods_mask = ~(~frame.isna()).sum(axis=1) >= min_periods
    result[insuff_periods_mask] = np.nan
    f_result = f_views.max(axis=(2, 3)).squeeze()

    d_out = pd.Series(np.nan, index=data[field].to_pandas().index)
    d_out[window - 1 :] = result
    if center:
        d_out = d_out.shift(-int(np.floor(window / 2)))

    f_out = pd.Series(np.nan, index=data[field].to_pandas().index)
    f_out[window - 1 :] = f_result
    if center:
        f_out = f_out.shift(-int(np.floor(window / 2)))

    data[target] = d_out
    flags[target] = f_out

    return data, flags


def _roll(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    func: Callable[[pd.Series], np.ndarray] | str = "mean",
    min_periods: int = 0,
    center: bool = True,
    **kwargs,
):
    validateFuncSelection(func, allow_operator_str=True)
    validateWindow(window)
    validateMinPeriods(min_periods)

    to_fit = data[field].copy()
    flags_col = flags[field].copy()
    if to_fit.empty:
        return data, flags

    d_roller = to_fit.rolling(window, min_periods=min_periods, center=center)
    if isinstance(func, str):
        to_fit = getattr(d_roller, func)()
    else:
        to_fit = d_roller.apply(func)

    flags_col = flags_col.rolling(window, min_periods=min_periods, center=center).max()
    data[field] = to_fit
    flags[field] = flags_col
    return data, flags
