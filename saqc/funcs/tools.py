#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import Literal

from saqc import FILTER_NONE, UNFLAGGED
from saqc.core import processing, register
from saqc.lib.checking import validateChoice
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
        validateChoice(mode, "mode", ["periodic", "selection_field"])

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

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        multivariate=True,
    )
    def plot(
        self: "SaQC",
        field: str | list[str],
        path: str | None = None,
        max_gap: str | None = None,
        mode: Literal["subplots", "oneplot"] | str = "oneplot",
        history: Literal["valid", "complete"] | list[str] | None = "valid",
        xscope: slice | None = None,
        store_kwargs: dict | None = None,
        ax: mpl.axes.Axes | None = None,
        ax_kwargs: dict | None = None,
        marker_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
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

        mode :
           How to process multiple variables to be plotted:
           * `"oneplot"` : plot all variables with their flags in one axis (default)
           * `"subplots"` : generate subplot grid where each axis contains one variable plot with associated flags
           * `"biplot"` : plotting first and second variable in field against each other in a scatter plot  (point cloud).

        history :
            Discriminate the plotted flags with respect to the tests they originate from.

            * ``"valid"``: Only plot flags, that are not overwritten by subsequent tests.
              Only list tests in the legend, that actually contributed flags to the overall
              result.
            * ``None``: Just plot the resulting flags for one variable, without any historical
              and/or meta information.
            * list of strings: List of tests. Plot flags from the given tests, only.
            * ``complete`` (not recommended, deprecated): Plot all the flags set by any test, independently from them being removed or modified by
              subsequent modifications. (this means: plotted flags do not necessarily match with flags ultimately
              assigned to the data)

        xscope :
            Determine a chunk of the data to be plotted. ``xscope`` can be anything,
            that is a valid argument to the ``pandas.Series.__getitem__`` method.

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
            Axis keywords. Change axis specifics. Those are passed on to the
            `matplotlib.axes.Axes.set <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html>`_
            method and can have the options listed there.
            The following options are `saqc` specific:

            * ``"xlabel"``: Either single string, that is to be attached to all x-axis´, or
              a List of labels, matching the number of variables to plot in length, or a dictionary, directly
              assigning labels to certain fields - defaults to ``None`` (no labels)
            * ``"ylabel"``: Either single string, that is to be attached to all y-axis´, or
              a List of labels, matching the number of variables to plot in length, or a dictionary, directly
              assigning labels to certain fields - defaults to ``None`` (no labels)
            * ``"title"``: Either a List of labels, matching the number of variables to plot in length, or a dictionary, directly
              assigning labels to certain variables - defaults to ``None`` (every plot gets titled the plotted variables name)
            * ``"fontsize"``: (float) Adjust labeling and titeling fontsize
            * ``"nrows"``, ``"ncols"``: shape of the subplot matrix the plots go into: If both are assigned, a subplot
              matrix of shape `nrows` x `ncols` is generated. If only one is assigned, the unassigned dimension is 1.
              defaults to plotting into subplot matrix with 2 columns and the necessary number of rows to fit the
              number of variables to plot.

        marker_kwargs :
            Keywords to modify flags marker appearance. The markers are set via the
            `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_
            method and can have the options listed there.
            The following options are `saqc` specific:

            * ``"cycleskip"``: (int) start the cycle of shapes that are assigned any flag-type with a certain lag - defaults to ``0`` (no skip)

        plot_kwargs :
            Keywords to modify the plot appearance. The plotting is delegated to
            `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_, all options listed there are available. Additionally the following saqc specific configurations are possible:

            * ``"alpha"``: Either a scalar float in *[0,1]*, that determines all plots' transparencies, or
              a list of floats, matching the number of variables to plot.

            * ``"linewidth"``: Either single float in *[0,1]*, that determines the thickness of all plotted,
              or a list of floats, matching the number of variables to plot.



        Notes
        -----

        * Check/modify the module parameter `saqc.lib.plotting.SCATTER_KWARGS` to see/modify global marker defaults
        * Check/modify the module parameter `saqc.lib.plotting.PLOT_KWARGS` to see/modify global plot line defaults
        """
        if history == "complete":
            warnings.warn(
                "Plotting with history='complete' is deprecated and will be removed in a future release (2.5)."
                "To get access to an saqc variables complete flagging history and analyze or plot it in detail, use flags"
                "history acces via `qc._flags.history[variable_name].hist` and a plotting library, such as pyplot.\n"
                "Minimal Pseudo example, having a saqc.SaQC instance `qc`, holding a variable `'data1'`, "
                "and having matplotlib.pyplot imported as `plt`:\n\n"
                "plt.plot(data)\n"
                "for f in qc._flags.history['data1'].hist \n"
                "    markers = qc._flags.history['data1'].hist[f] > level \n"
                "    markers=data[markers] \n"
                "    plt.scatter(markers.index, markers.values) \n",
                DeprecationWarning,
            )

        if "phaseplot" in kwargs:
            warnings.warn(
                'Parameter "phaseplot" is deprecated and will be removed in a future release (2.5). Assign to parameter "mode" instead. (plot(field, mode=phaseplot))',
                DeprecationWarning,
            )
            mode = kwargs["phaseplot"]

        if "cycleskip" in (ax_kwargs or {}):
            warnings.warn(
                'Passing "cycleskip" option with the "ax_kwargs" parameter is deprecated and will be removed in a future release (2.5). '
                'The option now has to be passed with the "marker_kwargs" parameter',
                DeprecationWarning,
            )
            marker_kwargs["cycleskip"] = ax_kwargs.pop("cycleskip")

        data, flags = self._data.copy(), self._flags.copy()

        level = kwargs.get("flag", UNFLAGGED)

        if dfilter < np.inf:
            for f in field:
                data[f].loc[flags[f] >= dfilter] = np.nan

        store_kwargs = store_kwargs or {}
        ax_kwargs = ax_kwargs or {}
        marker_kwargs = marker_kwargs or {}
        plot_kwargs = plot_kwargs or {}

        if not path:
            mpl.use(_MPL_DEFAULT_BACKEND)
        else:
            mpl.use("Agg")

        fig = makeFig(
            data=data,
            field=field,
            flags=flags,
            level=level,
            mode=mode,
            max_gap=max_gap,
            history=history,
            xscope=xscope,
            ax=ax,
            ax_kwargs=ax_kwargs,
            scatter_kwargs=marker_kwargs,
            plot_kwargs=plot_kwargs,
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
