#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import UNFLAGGED
from saqc.core import register
from saqc.core.history import History
from saqc.lib.tools import isflagged
from saqc.lib.ts_operators import interpolateNANs, shift2Freq

if TYPE_CHECKING:
    from saqc import SaQC


# TODO: remove, when `interpolateIndex` and `interpolateInvalid are removed`
INTERPOLATION_METHODS = Literal[
    "linear",
    "time",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
]


def _resampleOverlapping(data: pd.Series, freq: str, fill_value):
    """TODO: docstring needed"""
    dtype = data.dtype
    end = data.index[-1].ceil(freq)
    data = data.resample(freq).max()
    data = data.combine(data.shift(1, fill_value=fill_value), max)
    if end not in data:
        data.loc[end] = fill_value
    return data.fillna(fill_value).astype(dtype)


class InterpolationMixin:
    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateByRolling(
        self: "SaQC",
        field: str,
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        """
        Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.

        Parameters
        ----------
        window :
            The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
            an string is interpreted as an offset. ( see `pandas.rolling` for more information).
            Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.

        func : default median
            The function used for aggregation.

        center :
            Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.

        min_periods :
            Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
            computed.
        """
        datcol = self._data[field]
        roller = datcol.rolling(window=window, center=center, min_periods=min_periods)
        try:
            func_name = func.__name__
            if func_name[:3] == "nan":
                func_name = func_name[3:]
            rolled = getattr(roller, func_name)()
        except AttributeError:
            rolled = roller.apply(func)

        na_mask = datcol.isna()
        interpolated = na_mask & rolled.notna()
        datcol[na_mask] = rolled[na_mask]
        self._data[field] = datcol

        flagcol = pd.Series(np.nan, index=self._flags[field].index)
        flagcol.loc[interpolated] = np.nan if flag is None else flag

        # todo kwargs must have all passed args except data,field,flags
        meta = {
            "func": "interpolateByRolling",
            "args": (field,),
            "kwargs": {
                "window": window,
                "func": func,
                "center": center,
                "min_periods": min_periods,
                "flag": flag,
                **kwargs,
            },
        }
        self._flags.history[field].append(flagcol, meta)

        return self

    @register(
        mask=["field"],
        demask=[],
        squeeze=[],  # func handles history by itself
    )
    def interpolate(
        self: "SaQC",
        field: str,
        method: INTERPOLATION_METHODS = "time",
        order: int = 2,
        limit: int | str | None = None,
        extrapolate: Literal["forward", "backward", "both"] | None = None,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        """
        Fill NaN and flagged values using an interpolation method.

        Parameters
        ----------
        method :
            Interpolation technique to use. One of:

            * ‘linear’: Ignore the index and treat the values as equally spaced.
            * ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
            * ‘index’, ‘values’: Use the actual numerical values of the index.
            * ‘pad’: Fill in NaNs using existing values.
            * ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’:
                 Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index.
                 Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g.
                 ``qc.interpolate(method='polynomial', order=5)``.
            * ‘krogh’, ‘spline’, ‘pchip’, ‘akima’, ‘cubicspline’:
                 Wrappers around the SciPy interpolation methods of similar names.
            * ‘from_derivatives’: Refers to scipy.interpolate.BPoly.from_derivatives

        order :
            Order of the interpolation method, ignored if not supported by the chosen ``method``

        limit :
            Maximum number of missing values to interpolate. Only gaps smaller than ``limit`` will be filled.
            The gap size can be given as a number of values (integer) or a temporal extensions (offset string).
            With ``None``, all missing values will be interpolated.

        extrapolate :
            Use parameter to perform extrapolation instead of interpolation onto the trailing and/or leading chunks of
            NaN values in data series.

            * 'None' (default) - perform interpolation
            * 'forward'/'backward' - perform forward/backward extrapolation
            * 'both' - perform forward and backward extrapolation

        Examples
        --------
        See some examples of the keyword interplay below:

        Lets generate some dummy data:

        .. doctest:: interpolate

           >>> data = pd.DataFrame({'data':np.array([np.nan, 0, np.nan, np.nan, np.nan, 4, 5, np.nan, np.nan, 8, 9, np.nan, np.nan])}, index=pd.date_range('2000',freq='1H', periods=13))
           >>> data
                                data
           2000-01-01 00:00:00   NaN
           2000-01-01 01:00:00   0.0
           2000-01-01 02:00:00   NaN
           2000-01-01 03:00:00   NaN
           2000-01-01 04:00:00   NaN
           2000-01-01 05:00:00   4.0
           2000-01-01 06:00:00   5.0
           2000-01-01 07:00:00   NaN
           2000-01-01 08:00:00   NaN
           2000-01-01 09:00:00   8.0
           2000-01-01 10:00:00   9.0
           2000-01-01 11:00:00   NaN
           2000-01-01 12:00:00   NaN

        Use :py:meth:`~saqc.SaQC.interpolate` to do linear interpolation of up to 2 consecutive missing values:

        .. doctest:: interpolate

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.interpolate("data", limit=3, method='time')
           >>> qc.data # doctest:+NORMALIZE_WHITESPACE
                               data |
           ======================== |
           2000-01-01 00:00:00  NaN |
           2000-01-01 01:00:00  0.0 |
           2000-01-01 02:00:00  NaN |
           2000-01-01 03:00:00  NaN |
           2000-01-01 04:00:00  NaN |
           2000-01-01 05:00:00  4.0 |
           2000-01-01 06:00:00  5.0 |
           2000-01-01 07:00:00  6.0 |
           2000-01-01 08:00:00  7.0 |
           2000-01-01 09:00:00  8.0 |
           2000-01-01 10:00:00  9.0 |
           2000-01-01 11:00:00  NaN |
           2000-01-01 12:00:00  NaN |
           <BLANKLINE>


        Use :py:meth:`~saqc.SaQC.interpolate` to do linear extrapolaiton of up to 1 consecutive missing values:

        .. doctest:: interpolate

           >>> qc = saqc.SaQC(data)
           >>> qc = qc.interpolate("data", limit=2, method='time', extrapolate='both')
           >>> qc.data # doctest:+NORMALIZE_WHITESPACE
                               data |
           ======================== |
           2000-01-01 00:00:00  0.0 |
           2000-01-01 01:00:00  0.0 |
           2000-01-01 02:00:00  NaN |
           2000-01-01 03:00:00  NaN |
           2000-01-01 04:00:00  NaN |
           2000-01-01 05:00:00  4.0 |
           2000-01-01 06:00:00  5.0 |
           2000-01-01 07:00:00  NaN |
           2000-01-01 08:00:00  NaN |
           2000-01-01 09:00:00  8.0 |
           2000-01-01 10:00:00  9.0 |
           2000-01-01 11:00:00  NaN |
           2000-01-01 12:00:00  NaN |
           <BLANKLINE>
        """

        if "freq" in kwargs:
            # the old interpolate version
            warnings.warn(
                f"""
                The method `intepolate` is deprecated and will be removed in version 3.0 of saqc.
                To achieve the same behaviour please use:
                `qc.align(field={field}, freq={kwargs["freq"]}, method={method}, order={order}, flag={flag})`
                """,
                DeprecationWarning,
            )
            return self.align(
                field=field,
                freq=kwargs.pop("freq", method),
                method=method,
                order=order,
                flag=flag,
                **kwargs,
            )

        inter_data = interpolateNANs(
            self._data[field],
            method,
            order=order,
            gap_limit=limit,
            extrapolate=extrapolate,
        )

        interpolated = self._data[field].isna() & inter_data.notna()
        self._data[field] = inter_data
        new_col = pd.Series(np.nan, index=self._flags[field].index)
        new_col.loc[interpolated] = np.nan if flag is None else flag

        # todo kwargs must have all passed args except data,field,flags
        self._flags.history[field].append(
            new_col, {"func": "interpolateInvalid", "args": (), "kwargs": kwargs}
        )

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def align(
        self: "SaQC",
        field: str,
        freq: str,
        method: INTERPOLATION_METHODS = "time",
        order: int = 2,
        extrapolate: Literal["forward", "backward", "both"] | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Convert time series to specified frequency. Values affected by frequency
        changes will be inteprolated using the given method.

        Parameters
        ----------
        freq :
            Target frequency.

        method :
            Interpolation technique to use. One of:

            * ``'nshift'``: shift grid points to the nearest time stamp in the range = +/- 0.5 * ``freq``
            * ``'bshift'``: shift grid points to the first succeeding time stamp (if any)
            * ``'fshift'``: shift grid points to the last preceeding time stamp (if any)
            * ``'linear'``: Ignore the index and treat the values as equally spaced.
            * ``'time'``, ``'index'``, 'values': Use the actual numerical values of the index.
            * ``'pad'``: Fill in NaNs using existing values.
            * ``'nearest'``, ``'zero'``, ``'slinear'``, ``'quadratic'``, ``'cubic'``, ``'spline'``, ``'barycentric'``, ``'polynomial'``:
              Passed to ``scipy.interpolate.interp1d``. These methods use the numerical values of the index. Both ``'polynomial'`` and
              ``'spline'`` require that you also specify an ``order``, e.g. ``qc.interpolate(method='polynomial', order=5)``.
            * ``'krogh'``, ``'spline'``, ``'pchip'``, ``'akima'``, ``'cubicspline'``:
              Wrappers around the SciPy interpolation methods of similar names.
            * ``'from_derivatives'``: Refers to ``scipy.interpolate.BPoly.from_derivatives``

        order :
            Order of the interpolation method, ignored if not supported by the chosen ``method``

        extrapolate :
            Use parameter to perform extrapolation instead of interpolation onto the trailing and/or leading chunks of
            NaN values in data series.

            * ``None`` (default) - perform interpolation
            * ``'forward'``/``'backward'`` - perform forward/backward extrapolation
            * ``'both'`` - perform forward and backward extrapolation

        overwrite :
           If set to True, existing flags will be cleared
        """

        # TODO:
        # - should we keep `extrapolate`

        if self._data[field].empty:
            return self

        if method in ("fshift", "bshift", "nshift"):
            datacol, history = _shift(
                saqc=self, field=field, freq=freq, method=method, **kwargs
            )
        else:
            datacol, history = _interpolate(
                saqc=self,
                field=field,
                freq=freq,
                method=method,
                order=order,
                extrapolate=extrapolate,
                dfilter=kwargs["dfilter"],
            )

        meta = {
            "func": "align",
            "args": (field,),
            "kwargs": {
                "freq": freq,
                "method": method,
                "order": order,
                "extrapolate": extrapolate,
                **kwargs,
            },
        }

        flagcol = pd.Series(UNFLAGGED if overwrite else np.nan, index=history.index)
        history.append(flagcol, meta)

        self._data[field] = datacol
        self._flags.history[field] = history

        return self

    ### Deprecated functions

    @register(mask=["field"], demask=[], squeeze=[])
    def interpolateIndex(
        self: "SaQC",
        field: str,
        freq: str,
        method: INTERPOLATION_METHODS,
        order: int = 2,
        limit: int | None = 2,
        extrapolate: Literal["forward", "backward", "both"] = None,
        **kwargs,
    ) -> "SaQC":
        """
        Function to interpolate the data at regular (äquidistant) timestamps (or Grid points).

        Parameters
        ----------
        freq :
            An Offset String, interpreted as the frequency of
            the grid you want to interpolate your data to.

        method :
            The interpolation method you want to apply.

        order :
            If your selected interpolation method can be performed at different 'orders' - here you pass the desired
            order.

        limit :
            Upper limit of missing index values (with respect to ``freq``) to fill. The limit can either be expressed
            as the number of consecutive missing values (integer) or temporal extension of the gaps to be filled
            (Offset String).
            If ``None`` is passed, no limit is set.

        extraplate :
            Use parameter to perform extrapolation instead of interpolation onto the trailing and/or leading chunks of
            NaN values in data series.

            * ``None`` (default) - perform interpolation
            * ``'forward'``/``'backward'`` - perform forward/backward extrapolation
            * ``'both'`` - perform forward and backward extrapolation
        """

        msg = """
        The method `interpolateIndex` is deprecated and will be removed in verion 3.0 of saqc.
        To achieve the same behavior use:
        """
        call = "qc.align(field={field}, freq={freq}, method={method}, order={order}, extrapolate={extrapolate})"
        if limit != 2:
            call = f"{call}.interpolate(field={field}, method={method}, order={order}, limit={limit}, extrapolate={extrapolate})"

        warnings.warn(f"{msg}`{call}`", DeprecationWarning)
        out = self.align(
            field=field,
            freq=freq,
            method=method,
            order=order,
            extrapolate=extrapolate,
            **kwargs,
        )
        if limit != 2:
            out = out.interpolate(
                field=field,
                freq=freq,
                method=method,
                order=order,
                limit=limit,
                extrapolate=extrapolate,
                **kwargs,
            )
        return out

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateInvalid(
        self: "SaQC",
        field: str,
        method: INTERPOLATION_METHODS,
        order: int = 2,
        limit: int | None = None,
        extrapolate: Literal["forward", "backward", "both"] | None = None,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        warnings.warn(
            f"""
            The method `intepolateInvalid` is deprecated and will be removed
            with version 3.0 of saqc. To achieve the same behavior, please use
            `qc.interpolate(
                field={field}, method={method}, order={order},
                limit={limit}, extrapolate={extrapolate}, flag={flag}
            )`
            """
        )

        return self.interpolate(
            field=field,
            method=method,
            order=order,
            limit=limit,
            extrapolate=extrapolate,
            flag=flag,
            **kwargs,
        )


def _shift(
    saqc: "SaQC",
    field: str,
    freq: str,
    method: Literal["fshift", "bshift", "nshift"] = "nshift",
    **kwargs,
) -> Tuple[pd.Series, History]:
    """
    Shift data points and flags to a regular frequency grid.

    Parameters
    ----------
    field :
        The fieldname of the column, holding the data-to-be-shifted.

    freq :
        Offset string. Sampling rate of the target frequency.

    method :
        Method to propagate values:

        * 'nshift' : shift grid points to the nearest time stamp in the range = +/- 0.5 * ``freq``
        * 'bshift' : shift grid points to the first succeeding time stamp (if any)
        * 'fshift' : shift grid points to the last preceeding time stamp (if any)

    freq_check :
        * ``None`` : do not validate the ``freq`` string.
        * 'check' : check ``freq`` against an frequency estimation, produces a warning in case of miss matches.
        * 'auto' : estimate frequency, `freq` is ignored.

    Returns
    -------
    saqc.SaQC
    """
    # TODO
    # - Do we need `freq_check`? If so could we move it to `align`?

    datcol = saqc._data[field]
    if datcol.empty:
        return saqc

    # do the shift
    datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)

    # do the shift on the history
    kws = dict(method=method, freq=freq)

    history = saqc._flags.history[field].apply(
        index=datcol.index,
        func_handle_df=True,
        func=shift2Freq,
        func_kws={**kws, "fill_value": np.nan},
    )

    return datcol, history


def _interpolate(
    saqc: "SaQC",
    field: str,
    freq: str,
    method: str,
    order: int | None,
    dfilter: float,
    extrapolate: Literal["forward", "backward", "both"] | None = None,
) -> Tuple[pd.Series, History]:
    datcol = saqc._data[field].copy()

    start, end = datcol.index[0].floor(freq), datcol.index[-1].ceil(freq)
    grid_index = pd.date_range(start=start, end=end, freq=freq, name=datcol.index.name)

    flagged = isflagged(saqc._flags[field], dfilter)

    # drop all points that hold no relevant grid information
    datcol = datcol[~flagged].dropna()

    # account for annoying case of subsequent frequency aligned values,
    # that differ exactly by the margin of 2*freq
    gaps = datcol.index[1:] - datcol.index[:-1] == 2 * pd.Timedelta(freq)
    gaps = datcol.index[1:][gaps]
    gaps = gaps.intersection(grid_index).shift(-1, freq)

    # prepare grid interpolation:
    datcol = datcol.reindex(datcol.index.union(grid_index))

    # do the grid interpolation
    inter_data = interpolateNANs(
        data=datcol,
        method=method,
        order=order,
        gap_limit=2,
        extrapolate=extrapolate,
    )
    # override falsely interpolatet values:
    inter_data[gaps] = np.nan
    inter_data = inter_data[grid_index]

    history = saqc._flags.history[field].apply(
        index=inter_data.index,
        func=_resampleOverlapping,
        func_kws=dict(freq=freq, fill_value=np.nan),
    )
    return inter_data, history
