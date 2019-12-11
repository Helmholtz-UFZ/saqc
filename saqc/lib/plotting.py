#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np


__plotvars = []

_colors = dict(unflagged='silver', good='seagreen', bad='firebrick', suspicious='gold')


def plotAllHook(data, flagger):
    if len(__plotvars) > 1:
        _plot(data, flagger, True, __plotvars)


def plotHook(data, old, new, varname, do_plot, flag_test, plot_nans=False):
    # old/new: flagger
    if do_plot:
        __plotvars.append(varname)
        # cannot use getFlags here, because if a flag was set (e.g. with force) the
        # flag may be the same, but any additional row (e.g. comment-field) would differ
        mask = (old._flags[varname] != new._flags[varname])
        if isinstance(mask, pd.DataFrame):
            mask = mask.any(axis=1)
        _plot(data, new, mask, varname, title=flag_test, plot_nans=plot_nans)


def _plot(
    data,
    flagger,
    flagmask,
    varname,
    interactive_backend=True,
    title="Data Plot",
    plot_nans=True,
):

    # todo: try catch warn (once) return
    # only import if plotting is requested by the user
    import matplotlib as mpl

    if not interactive_backend:
        # Import plot libs without interactivity, if not needed. This ensures that this can
        # produce an plot.png even if tkinter is not installed. E.g. if one want to run this
        # on machines without X-Server aka. graphic interface.
        mpl.use("Agg")
    else:
        mpl.use("TkAgg")
    from matplotlib import pyplot as plt

    # needed for datetime conversion
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if not isinstance(varname, (list, set)):
        varname = [varname]
    varname = set(varname)

    # filter out variables to which no data is associated (e.g. freshly assigned vars)
    tmp = []
    for var in varname:
        if var in data.columns:
            tmp.append(var)
        else:
            logging.warning(f"Cannot plot column '{var}', because it is not present in data.")
    if not tmp:
        return
    varname = tmp

    plots = len(varname)
    if plots > 1:
        fig, axes = plt.subplots(plots, 1, sharex=True)
        axes[0].set_title(title)
        for i, v in enumerate(varname):
            _plotByQualtyFlag(data, v, flagger, flagmask, axes[i], plot_nans)
    else:
        fig, ax = plt.subplots()
        plt.title(title)
        _plotByQualtyFlag(data, varname.pop(), flagger, flagmask, ax, plot_nans)

    # dummy plot for the label `missing` see plot_vline for more info
    plt.plot([], [], ":", color="silver", label="missing data")

    plt.xlabel("time")
    plt.legend()

    if interactive_backend:
        plt.show()


def _plotByQualtyFlag(data, varname, flagger, flagmask, ax, plot_nans):
    ax.set_ylabel(varname)

    x = data.index
    y = data[varname]

    # base plot: show all(!) data
    ax.plot(x, y, "-", color="silver", label="data")

    # ANY OLD FLAG
    # plot all(!) data that are already flagged in black
    flagged = flagger.isFlagged(varname, flag=flagger.GOOD, comparator='>=')
    oldflags = flagged & ~flagmask
    ax.plot(x[oldflags], y[oldflags], ".", color="black", label="flagged by other test")
    if plot_nans:
        _plot_nans(y[oldflags], 'black', ax)

    # now we just want to show data that was flagged
    if flagmask is not True:
        x = x[flagmask]
        y = y[flagmask]
        flagger = flagger.getFlagger(varname, flagmask)

    if x.empty:
        return

    suspicious = pd.Series(data=np.ones(len(y), dtype=bool), index=y.index)
    # flag by categories

    # plot UNFLAGGED (only nans are needed)
    flag, color = flagger.UNFLAGGED, _colors['unflagged']
    flagged = flagger.isFlagged(varname, flag=flag, comparator='==')
    ax.plot(x[flagged], y[flagged], '.', color=color, label=f"flag: {flag}")
    if plot_nans:
        _plot_nans(y[flagged], color, ax)

    # plot GOOD
    flag, color = flagger.GOOD, _colors['good']
    flagged = flagger.isFlagged(varname, flag=flag, comparator='==')
    ax.plot(x[flagged], y[flagged], '.', color=color, label=f"flag: {flag}")
    if plot_nans:
        _plot_nans(y[flagged], color, ax)

    # plot BAD
    flag, color = flagger.BAD, _colors['bad']
    flagged = flagger.isFlagged(varname, flag=flag, comparator='==')
    ax.plot(x[flagged], y[flagged], '.', color=color, label=f"flag: {flag}")
    if plot_nans:
        _plot_nans(y[flagged], color, ax)

    # plot SUSPICIOS
    color = _colors['suspicious']
    flagged = flagger.isFlagged(varname, flag=flagger.GOOD, comparator='>')
    flagged &= flagger.isFlagged(varname, flag=flagger.BAD, comparator='<')
    ax.plot(x[flagged], y[flagged], '.', color=color, label=f"{flagger.GOOD} < flag < {flagger.BAD}")
    if plot_nans:
        _plot_nans(y[flagged], color, ax)


def _plot_nans(y, color, ax):
    nans = y.isna()
    _plotVline(ax, y[nans].index, color=color)


def _plotVline(plt, points, color="blue"):
    # workaround for ax.vlines() as this work unexpected
    # normally this should work like so:
    #   ax.vlines(idx, *ylim, linestyles=':', color='silver', label="missing")
    for point in points:
        plt.axvline(point, color=color, linestyle=":")

