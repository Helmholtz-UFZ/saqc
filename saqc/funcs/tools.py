#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import Literal

from saqc import FILTER_NONE, UNFLAGGED
from saqc.core import processing, register
from saqc.lib.docs import DOC_TEMPLATES
from saqc.lib.plotting import makeFig
from saqc.lib.tools import periodicMask

if TYPE_CHECKING:
    from saqc import SaQC


_MPL_DEFAULT_BACKEND = mpl.get_backend()


class ToolsMixin:
    @register(
        mask=[],
        demask=[],
        squeeze=[],
        handles_target=True,
        docstring={"target": DOC_TEMPLATES["target"]},
    )
    def copyField(
        self: "SaQC",
        field: str,
        target: str,
        overwrite: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Copy data and flags to a new name (preserve flags history).
        """
        if field == target:
            return self

        if target in self._flags.columns.union(self._data.columns):
            if not overwrite:
                raise ValueError(f"{target}: already exist")
            self = self.dropField(field=target)

        self._data[target] = self._data[field].copy()
        self._flags.history[target] = self._flags.history[field].copy()

        return self

    @processing()
    def dropField(self: "SaQC", field: str, **kwargs) -> "SaQC":
        """
        Drops field from the data and flags.
        """
        del self._data[field]
        del self._flags[field]
        return self

    @processing()
    def renameField(self: "SaQC", field: str, new_name: str, **kwargs) -> "SaQC":
        """
        Rename field in data and flags.

        Parameters
        ----------
        new_name :
            String, field is to be replaced with.
        """
        self._data[new_name] = self._data[field]
        self._flags.history[new_name] = self._flags.history[field]
        del self._data[field]
        del self._flags[field]
        return self

    @register(mask=[], demask=[], squeeze=["field"])
    def selectTime(
        self: "SaQC",
        field: str,
        mode: Literal["periodic", "selection_field"],
        selection_field: str | None = None,
        start: str | None = None,
        end: str | None = None,
        closed: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Realizes masking within saqc.

        Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
        values or datachunks from flagging routines. This function replaces flags with UNFLAGGED
        value, wherever values are to get masked. Furthermore, the masked values get replaced by
        np.nan, so that they dont effect calculations.

        Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:

        1. dublicate "field" in the input data (`copyField`)
        2. mask the dublicated data (this, `selectTime`)
        3. apply the tests you only want to be applied onto the masked data chunks (a saqc function)
        4. project the flags, calculated on the dublicated and masked data onto the original field data
            (`concateFlags` or `flagGeneric`)
        5. drop the dublicated data (`dropField`)

        To see an implemented example, checkout flagSeasonalRange in the saqc.functions module

        Parameters
        ----------
        mode :
            The masking mode.
            - "periodic": parameters "period_start", "end" are evaluated to generate a periodical mask
            - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.

        selection_field :
            Only effective if mode == "mask_var"
            Fieldname of the column, holding the data that is to be used as mask. (must be boolean series)
            Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
            indices will be calculated and values get masked where the values of the inner join are ``True``.

        start :
            Only effective if mode == "seasonal"
            String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
            Has to be of same length as `end` parameter.
            See examples section below for some examples.

        end :
            Only effective if mode == "periodic"
            String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
            Has to be of same length as `end` parameter.
            See examples section below for some examples.

        closed :
            Wheather or not to include the mask defining bounds to the mask.

        Examples
        --------
        The `period_start` and `end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
        They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
        (mm=month, dd=day, HH=hour, MM=minute, SS=second)
        Single digit specifications have to be given with leading zeros.
        `period_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
        The highest date unit gives the period.
        For example:

        >>> start = "01T15:00:00"
        >>> end = "13T17:30:00"

        Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked

        >>> start = "01:00"
        >>> end = "04:00"

        All the values between the first and 4th minute of every hour get masked.

        >>> start = "01-01T00:00:00"
        >>> end = "01-03T00:00:00"

        Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
        include 00:00:00 at the first of march. To exclude this one, pass:

        >>> start = "01-01T00:00:00"
        >>> end = "02-28T23:59:59"

        To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
        season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:

        >>> start = "22:00:00"
        >>> end = "06:00:00"
        """
        datcol_idx = self._data[field].index

        if mode == "periodic":
            mask = periodicMask(datcol_idx, start, end, ~closed)
        elif mode == "selection_field":
            idx = self._data[selection_field].index.intersection(datcol_idx)
            mask = self._data[selection_field].loc[idx]
        else:
            raise ValueError(
                "Keyword passed as masking mode is unknown ({})!".format(mode)
            )

        mask = mask.reindex(self._data[field].index, fill_value=False).astype(bool)
        self._data[field].loc[mask] = np.nan
        self._flags[mask, field] = UNFLAGGED
        return self

    @register(mask=[], demask=[], squeeze=[])
    def plot(
        self: "SaQC",
        field: str,
        path: str | None = None,
        max_gap: str | None = None,
        history: Literal["valid", "complete"] | list[str] | None = "valid",
        xscope: slice | None = None,
        phaseplot: str | None = None,
        store_kwargs: dict | None = None,
        ax: mpl.axes.Axes | None = None,
        ax_kwargs: dict | None = None,
        dfilter: float = FILTER_NONE,
        **kwargs,
    ) -> "SaQC":
        """
        Plot data and flags or store plot to file.

        There are two modes, 'interactive' and 'store', which are determined through the
        ``save_path`` keyword. In interactive mode (default) the plot is shown at runtime
        and the program execution stops until the plot window is closed manually. In
        store mode the generated plot is stored to disk and no manually interaction is
        needed.

        Parameters
        ----------
        path :
            If ``None`` is passed, interactive mode is entered; plots are shown immediatly
            and a user need to close them manually before execution continues.
            If a filepath is passed instead, store-mode is entered and
            the plot is stored unter the passed location.

        max_gap :
            If ``None``, all data points will be connected, resulting in long linear
            lines, in case of large data gaps. ``NaN`` values will be removed before
            plotting. If an offset string is passed, only points that have a distance
            below ``max_gap`` are connected via the plotting line.

        history :
            Discriminate the plotted flags with respect to the tests they originate from.

            * ``"valid"``: Only plot flags, that are not overwritten by subsequent tests.
              Only list tests in the legend, that actually contributed flags to the overall
              result.
            * ``"complete"``: Plot all flags set and list all the tests executed on a variable.
              Suitable for debugging/tracking.
            * ``None``: Just plot the resulting flags for one variable, without any historical
              and/or meta information.
            * list of strings: List of tests. Plot flags from the given tests, only.

        xscope :
            Determine a chunk of the data to be plotted processed. ``xscope`` can be anything,
            that is a valid argument to the ``pandas.Series.__getitem__`` method.

        phaseplot :
            If a string is passed, plot ``field`` in the phase space it forms together with the
            variable ``phaseplot``.

        ax :
            If not ``None``, plot into the given ``matplotlib.Axes`` instance, instead of a
            newly created ``matplotlib.Figure``. This option offers a possibility to integrate
            ``SaQC`` plots into custom figure layouts.

        store_kwargs :
            Keywords to be passed on to the ``matplotlib.pyplot.savefig`` method, handling
            the figure storing. To store an pickle object of the figure, use the option
            ``{"pickle": True}``, but note that all other ``store_kwargs`` are ignored then.
            To reopen a pickled figure execute: ``pickle.load(open(savepath, "w")).show()``

        ax_kwargs :
            Axis keywords. Change the axis labeling defaults. Most important keywords:
            ``"xlabel"``, ``"ylabel"``, ``"title"``, ``"fontsize"``, ``"cycleskip"``.
        """
        data, flags = self._data.copy(), self._flags.copy()

        level = kwargs.get("flag", UNFLAGGED)

        if dfilter < np.inf:
            data[field].loc[flags[field] >= dfilter] = np.nan

        if store_kwargs is None:
            store_kwargs = {}

        if ax_kwargs is None:
            ax_kwargs = {}

        if not path:
            mpl.use(_MPL_DEFAULT_BACKEND)
        else:
            mpl.use("Agg")

        fig = makeFig(
            data=data,
            field=field,
            flags=flags,
            level=level,
            max_gap=max_gap,
            history=history,
            xscope=xscope,
            phaseplot=phaseplot,
            ax=ax,
            ax_kwargs=ax_kwargs,
        )

        if ax is None and not path:
            plt.show()

        if path:
            if store_kwargs.pop("pickle", False):
                with open(path, "wb") as f:
                    pickle.dump(fig, f)
            else:
                fig.savefig(path, **store_kwargs)

        return self
