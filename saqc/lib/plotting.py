#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd

from saqc.flagger import BaseFlagger


logger = logging.getLogger("SaQC")

__plotvars = []

_colors = {
    "unflagged": "silver",
    "good": "seagreen",
    "bad": "firebrick",
    "suspicious": "gold",
}

_figsize = (10, 4)


def plotAllHook(data, flagger, plot_nans=False):
    if __plotvars:
        _plot(data, flagger, True, __plotvars, plot_nans=plot_nans)


def plotHook(
    data: pd.DataFrame,
    flagger_old: BaseFlagger,
    flagger_new: BaseFlagger,
    varname: str,
    flag_test: str,
    plot_nans: bool = False,
):

    # if data was harmonized, nans may occur in flags
    harm_nans = flagger_new.getFlags(varname).isna() | flagger_old.getFlags(varname).isna()

    # clean data from harmonisation nans
    if harm_nans.any():
        data = data.loc[~harm_nans, varname]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # clean flags from harmonisation nans
    try:
        flagger_old = flagger_old.getFlagger(varname, loc=data.index)
    except ValueError:
        # this might fail if someone want to plot the harmonisation itself,
        # but then we just plot the 'new' flags, ignoring the diff to the old ones
        mask = True
    else:
        flagger_new = flagger_new.getFlagger(varname, loc=data.index)

        # cannot use getFlags here, because if a flag was set (e.g. with force) the
        # flag may be the same, but any additional row (e.g. comment-field) would differ
        flags_old = flagger_old._flags[varname]
        flags_new = flagger_new._flags[varname]
        if len(flags_old) != len(flags_new):
            # NOTE:
            # we are just getting the result of an
            # harmonization, nothing to see here
            return

        mask = flags_old != flags_new
        if isinstance(mask, pd.DataFrame):
            mask = mask.any(axis=1)

    __plotvars.append(varname)
    _plot(data, flagger_new, mask, varname, title=flag_test, plot_nans=plot_nans)


def _plot(
    data, flagger, flagmask, varname, interactive_backend=True, title="Data Plot", plot_nans=True,
):

    # todo: try catch warn (once) return
    # only import if plotting is requested by the user
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters

    # needed for datetime conversion
    register_matplotlib_converters()

    if not interactive_backend:
        # Import plot libs without interactivity, if not needed. This ensures that this can
        # produce an plot.png even if tkinter is not installed. E.g. if one want to run this
        # on machines without X-Server aka. graphic interface.
        mpl.use("Agg")
    else:
        mpl.use("TkAgg")

    if not isinstance(varname, (list, set)):
        varname = [varname]
    varname = set(varname)

    # filter out variables to which no data is associated (e.g. freshly assigned vars)
    tmp = []
    for var in varname:
        if var in data.columns:
            tmp.append(var)
        else:
            logger.warning(f"Cannot plot column '{var}', because it is not present in data.")
    if not tmp:
        return
    varnames = tmp

    plots = len(varnames)
    if plots > 1:
        fig, axes = plt.subplots(plots, 1, sharex=True, figsize=_figsize)
        axes[0].set_title(title)
        for i, v in enumerate(varnames):
            _plotByQualityFlag(data, v, flagger, flagmask, axes[i], plot_nans)
    else:
        fig, ax = plt.subplots(figsize=_figsize)
        plt.title(title)
        _plotByQualityFlag(data, varnames.pop(), flagger, flagmask, ax, plot_nans)

    # dummy plot for the label `missing` see _plotVline for more info
    if plot_nans:
        plt.plot([], [], ":", color="silver", label="missing data")

    plt.xlabel("time")
    plt.legend()

    if interactive_backend:
        plt.show()


def _plotByQualityFlag(data, varname, flagger, flagmask, ax, plot_nans):
    ax.set_ylabel(varname)

    if flagmask is True:
        flagmask = pd.Series(data=np.ones(len(data), dtype=bool), index=data.index)

    data = data[varname]
    if not plot_nans:
        data = data.dropna()
        flagmask = flagmask.loc[data.index]

    flagger = flagger.getFlagger(varname, loc=data.index)

    # base plot: show all(!) data
    ax.plot(
        data,
        # NOTE: no lines to flagged points
        # data.index, np.ma.array(data.values, mask=flagger.isFlagged(varname).values),
        "-",
        color="silver",
        label="data",
    )

    # ANY OLD FLAG
    # plot all(!) data that are already flagged in black
    flagged = flagger.isFlagged(varname, flag=flagger.GOOD, comparator=">=")
    oldflags = flagged & ~flagmask

    ax.plot(data[oldflags], ".", color="black", label="flagged by other test")
    if plot_nans:
        _plotNans(data[oldflags], "black", ax)

    # now we just want to show data that was flagged
    data = data.loc[flagmask[flagmask].index]
    flagger = flagger.getFlagger(varname, loc=data.index)

    if data.empty:
        return

    plots = [
        (flagger.UNFLAGGED, _colors["unflagged"]),
        (flagger.GOOD, _colors["good"]),
        (flagger.BAD, _colors["bad"]),
    ]

    for flag, color in plots:
        flagged = flagger.isFlagged(varname, flag=flag, comparator="==")
        if not data[flagged].empty:
            ax.plot(data[flagged], ".", color=color, label=f"flag: {flag}")
        if plot_nans:
            _plotNans(data[flagged], color, ax)

    # plot SUSPICIOS
    color = _colors["suspicious"]
    flagged = flagger.isFlagged(varname, flag=flagger.GOOD, comparator=">")
    flagged &= flagger.isFlagged(varname, flag=flagger.BAD, comparator="<")
    if not data[flagged].empty:
        ax.plot(
            data[flagged], ".", color=color, label=f"{flagger.GOOD} < flag < {flagger.BAD}",
        )
    if plot_nans:
        _plotNans(data[flagged], color, ax)


def _plotNans(y, color, ax):
    nans = y.isna()
    _plotVline(ax, y[nans].index, color=color)


def _plotVline(ax, points, color="blue"):
    # workaround for ax.vlines() as this work unexpected
    # normally this should work like so:
    #   ax.vlines(idx, *ylim, linestyles=':', color='silver', label="missing")
    for point in points:
        ax.axvline(point, color=color, linestyle=":")
