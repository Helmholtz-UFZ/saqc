#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from saqc.core import DictOfSeries, Flags, register
from saqc.lib.tools import getFreqDelta, toSequence
from saqc.lib.ts_operators import (
    butterFilter,
    polyRoller,
    polyRollerIrregular,
    polyRollerNoMissing,
)
from saqc.lib.types import Float, FreqStr, Int, OffsetStr, SaQC, ValidatePublicMembers

DEFAULT_MOMENT = dict(
    pretrained_model_name_or_path="AutonLab/MOMENT-1-large", revision="main"
)

FILL_METHODS = Literal[
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
]


class CurvefitMixin(ValidatePublicMembers):
    @register(mask=["field"], demask=[], squeeze=[])
    def fitPolynomial(
        self: SaQC,
        field: str,
        window: OffsetStr | (Int >= 0),
        order: Int >= 1,
        min_periods: Int >= 0 = 0,
        **kwargs,
    ) -> SaQC:
        """
        Fit a polynomial model to the data.

        The fit is calculated by fitting a polynomial of degree `order` to a data slice
        of size `window`, centered on each timestamp. The result overwrites the field
        unless a target is specified.

        For regularly sampled data:

        * If missing values are rare or residuals for windows with missing values are
          not needed, performance can be increased by setting min_periods=window.
        * The initial and final ``window``//2 timestamps do not get fitted.
        * Each residual is assigned the worst flag present in the corresponding interval
          of the original data.

        Parameters
        ----------
        window : int or str
            Size of the fitting window. If an integer is passed, it represents the number
            of timestamps in each window. If an offset string is passed, it represents the
            window's temporal extent. The window is centered around the timestamp being fitted.
            For uniformly sampled data, an odd number of timestamps is always used to constitute a window (subtracted by 1,
            if the total is even).

        order : int
            Degree of the polynomial used for fitting.

        min_periods : int
            Minimum number of timestamps in a window required to perform the fit.
            Windows with fewer timestamps will produce NaNs. Passing 0 disables this
            check and may result in overfitting for sparse windows.
        """
        self._data, self._flags = _fitPolynomial(
            data=self._data,
            field=field,
            flags=self._flags,
            window=window,
            order=order,
            min_periods=min_periods,
            **kwargs,
        )
        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def fitLowpassFilter(
        self: SaQC,
        field: str,
        cutoff: (Float >= 0) | FreqStr,
        nyq: Float >= 0 = 0.5,
        filter_order: Int >= 1 = 2,
        fill_method: FILL_METHODS = "linear",
        **kwargs,
    ) -> SaQC:
        """
        Fits the data using the butterworth filter.

        Note
        ----
        The data is expected to be regularly sampled.

        Parameters
        ----------
        cutoff :
            The cutoff-frequency, either an offset freq string, or expressed in multiples of the sampling rate.

        nyq :
            The niquist-frequency. expressed in multiples if the sampling rate.

        fill_method :
            Fill method to be applied on the data before filtering (butterfilter cant
            handle ''np.nan''). See documentation of pandas.Series.interpolate method for
            details on the methods associated with the different keywords.
        """

        self._data[field] = butterFilter(
            self._data[field],
            cutoff=cutoff,
            nyq=nyq,
            filter_order=filter_order,
            fill_method=fill_method,
            filter_type="lowpass",
        )
        return self

    @register(mask=["field"], demask=[], squeeze=[], multivariate=True)
    def fitMomentFM(
        self: "SaQC",
        field: str | list[str],
        ratio: int = 4,
        context: int = 512,
        agg: Literal["center", "mean", "median", "std"] = "mean",
        model_spec: dict | None = None,
        **kwargs,
    ):
        """
        Fits the data by reconstructing it with the Moment Foundational Timeseries Model (MomentFM).

        The function applies MomentFM [1] in its reconstruction mode on a window of size `context`, striding through the data
        with step size `context`/`ratio`

        Parameters
        ----------
        ratio :
            The number of samples generated for any values reconstruction. Must be a divisor of `context`.
            Effectively controlls the stride-width of the reconstruction window through the data.

        context :
            size of the context window with regard to wich any value is reconstructed.

        agg :
            How to aggregate the different reconstructions for the same value.
            * 'center': use the value that was constructed in a window centering around the origin value
            * 'mean': assign the mean over all reconstructed values
            * 'median': assign the median over all reconstructed values
            * 'std': assign the standard deviation over all reconstructed values

        model_spec :
            Dictionary with the fields:
            * `pretrained_model_name_or_path`
            * `revision`

            Defaults to global Parameter `DEFAULT_MOMENT=dict(pretrained_model_name_or_path="AutonLab/MOMENT-1-large", revision="main"`


        Examples
        --------
        .. figure:: /resources/images/fitFMpic.png


        Notes
        -----
        [1] https://arxiv.org/abs/2402.03885
        [2] https://github.com/moment-timeseries-foundation-model/moment
        """

        _model_scope = 512

        model_spec = DEFAULT_MOMENT if model_spec is None else model_spec

        try:
            import torch
            from momentfm import MOMENTPipeline
            from momentfm.data.informer_dataset import InformerDataset
            from torch.utils.data import DataLoader
        except ImportError as e:
            raise ImportError(
                f"Foundational Timeseries Regressor requirements not sufficed:\n{e}\n"
                'Install the requirements manually or by pip installing "saqc[FM]"'
            ) from e

        if context > _model_scope:
            raise ValueError(
                f'Parameter "context" must not be greater {_model_scope}. Got {context}.'
            )
        if context % ratio > 0:
            raise ValueError(
                f'Parameter "ratio" must be a divisor of "context". Got context={context} and ratio={ratio} -> divides as: {context / ratio}.'
            )
        if agg not in ["mean", "median", "std", "center"]:
            raise ValueError(
                f'Parameter "agg" needs be one out of ["mean", "median", "std", "center"]. Got "agg": {agg}.'
            )

        field = toSequence(field)
        dat = self._data[field].to_pandas()
        # input mask, in case context is < 512
        _input_mask = np.ones(_model_scope)
        _input_mask[context:] = 0
        # model instance
        model = MOMENTPipeline.from_pretrained(
            **model_spec,
            model_kwargs={"task_name": "reconstruction"},
        )
        model.init()

        # derive rec window stride width from ratio
        stepsz = context // ratio
        # na mask that will tell the model where data values are missing
        _na_mask = ~(dat.isna().any(axis=1))
        # generate sliding view on na mask equaling model input-patch size
        na_mask = np.lib.stride_tricks.sliding_window_view(
            (_na_mask).values, window_shape=_model_scope, axis=0
        )
        # generate stack of model input patches from the data
        dv = np.lib.stride_tricks.sliding_window_view(
            dat.values, window_shape=(_model_scope, len(dat.columns)), axis=(0, 1)
        )
        dv = np.swapaxes(dv, -1, -2).squeeze(1).astype("float32")
        # filter input stacks to represent stepsz - sized reconstruction window stride
        dv = dv[::stepsz]
        na_mask = na_mask[::stepsz].astype(int)
        # mask values to achieve truncated samples (if context < 512)
        input_mask = np.ones(na_mask.shape)
        input_mask[:, :] = _input_mask
        # to torch
        dv = torch.tensor(dv)
        na_mask = torch.tensor(na_mask)
        input_mask = torch.tensor(input_mask)
        # get reconstruction for sample stack
        output = model(x_enc=dv, mask=na_mask, input_mask=input_mask)
        reconstruction = output.reconstruction.detach().cpu().numpy()
        reconstruction = reconstruction[:, :, _input_mask.astype(bool)]
        # derive number of reconstruction windows covering the same value
        partition_count = context // stepsz
        # aggregate overlapping reconstruction windows to 1-d data
        for ef in enumerate(dat.columns):
            rec_arr = np.zeros([dat.shape[0], partition_count]).astype(float)
            rec_arr[:] = np.nan
            count_arr = rec_arr.copy()
            # arange reconstruction windows in array, so that same array row refers to same reconstructed time
            for s in range(rec_arr.shape[1]):
                d = reconstruction[s::partition_count, ef[0]].squeeze().flatten()
                offset = s * stepsz
                d_cut = min(offset + len(d), rec_arr.shape[0]) - offset
                rec_arr[offset : offset + len(d), s] = d[:d_cut]
                count_arr[offset : offset + len(d), s] = np.abs(
                    np.arange(d_cut) % context - 0.5 * context
                )

            # aggregate the rows with selected aggregation
            if agg == "center":
                c_select = count_arr.argmin(axis=1)
                rec = rec_arr[np.arange(len(c_select)), c_select]
            else:
                rec = getattr(np, "nan" + agg)(rec_arr, axis=1)

            self._data[ef[1]] = pd.Series(rec, index=dat.index, name=ef[1])

        return self


def _fitPolynomial(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[int, str],
    order: int,
    min_periods: int = 0,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    # TODO: some (rather large) parts are functional similar to saqc.funcs.rolling.roll

    if data[field].empty:
        return data, flags

    to_fit = data[field].copy()
    regular = getFreqDelta(to_fit.index)
    if not regular:
        if isinstance(window, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                "sample series."
            )
        # get interval centers
        centers = to_fit.rolling(
            pd.Timedelta(window) / 2, closed="both", min_periods=min_periods
        ).count()
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        fitted = to_fit.rolling(
            pd.Timedelta(window), closed="both", min_periods=min_periods, center=True
        ).apply(polyRollerIrregular, args=(centers, order))
    else:  # if regular
        if isinstance(window, str):
            window = pd.Timedelta(window) // regular
        if window % 2 == 0:
            window = int(window - 1)
        if min_periods is None:
            min_periods = window

        val_range = np.arange(0, window)
        center_index = window // 2
        if min_periods < window:
            if min_periods > 0:
                to_fit = to_fit.rolling(
                    window, min_periods=min_periods, center=True
                ).apply(lambda x, y: x[y], raw=True, args=(center_index,))

            # we need a missing value marker that is not nan,
            # because nan values dont get passed by pandas rolling method
            miss_marker = to_fit.min()
            miss_marker = np.floor(miss_marker - 1)
            na_mask = to_fit.isna()
            to_fit[na_mask] = miss_marker

            fitted = to_fit.rolling(window, center=True).apply(
                polyRoller,
                args=(miss_marker, val_range, center_index, order),
                raw=True,
            )
            fitted[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            fitted = to_fit.rolling(window, center=True).apply(
                polyRollerNoMissing,
                args=(val_range, center_index, order),
                raw=True,
            )

    data[field] = fitted
    worst = flags[field].rolling(window, center=True, min_periods=min_periods).max()
    flags[field] = worst

    return data, flags
