#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import dios.dios as dios
import matplotlib.pyplot as plt
from typing import List

from saqc.flagger import BaseFlagger

import_done = False
__plotvars = []

_cols = [
    # reference
    "ref-flags",
    "ref-data",
    "ref-data-nans",
    # other
    "flag-nans",
    # data
    "data-line",
    "data",
    "data-nans",
    # flags
    "unflagged",
    "old-flags",
    "good",
    "suspicious",
    "bad",
]
_colors = {
    # flags
    "unflagged": dict(marker='.', linestyle='none', color="silver", label="UNFLAGGED"),
    "good": dict(marker='.', linestyle='none', color="seagreen", label="GOOD"),
    "bad": dict(marker='.', linestyle='none', color="firebrick", label="BAD"),
    "suspicious": dict(marker='.', linestyle='none', color="gold", label="SUSPICIOUS"),
    "old-flags": dict(marker='.', linestyle='none', color="black"),
    # data
    "data": dict(marker='.', color="silver", label="data"),
    "data-line": dict( color="silver", label="data"),
    "data-nans": dict(marker='o', fillstyle='none', linestyle='none', color="lightsteelblue", label="NaN"),
    # reference
    "ref-flags": dict(marker='.', linestyle='none', color="silver"),
    "ref-data": dict( color="silver"),
    "ref-data-nans": dict(marker='o', fillstyle='none', linestyle='none', color="silver", label="NaN"),
    # other
    "flag-nans": dict(marker='o', fillstyle='none', linestyle='none', color="lightsteelblue", label=""),
}

_figsize = (10, 4)


def __import_helper(ion=False):
    global import_done
    if import_done:
        return
    import_done = True

    import matplotlib as mpl
    from pandas.plotting import register_matplotlib_converters

    # needed for datetime conversion
    register_matplotlib_converters()

    if not ion:
        # Import plot libs without interactivity, if not needed. This ensures that this can
        # produce an plot.png even if tkinter is not installed. E.g. if one want to run this
        # on machines without X-Server aka. graphic interface.
        mpl.use("Agg")
    else:
        mpl.use("TkAgg")


def plotAllHook(data, flagger, plot_nans=False):
    if __plotvars:
        _plot(data, flagger, True, __plotvars, plot_nans=plot_nans)


def plotHook(
        data_old: dios.DictOfSeries,
        data_new: dios.DictOfSeries,
        flagger_old: BaseFlagger,
        flagger_new: BaseFlagger,
        varnames: List[str],
        plot_name: str,
        show_nans_as_cycles: bool = False,
):
    # todo:
    #   - new/changed data ?
    #       - new column -> new varname -> plot only new(+all flags)
    #       - changed data -> old != new -> plot new data(+all flags), old(no flags) as reference
    #       - index-change -> probably harmo -> plot new data(+all flags), old(no flags) as reference
    #   - else: split in old and new flags by diff (a!=b), plot data, old flags in black, other by color
    # todo:
    #  - assert var in datanew
    #  - var not in data -> assigned new -> flags-line
    #  - old.index != new.index -> (i)  harmo -> 2 plots
    #
    __import_helper(ion=True)

    if len(varnames) != 1:
        NotImplementedError("currently only single changed variables can be plotted")
    var = varnames[0]

    assert var in flagger_new.flags
    flags = flagger_new.flags[var]
    flags: pd.Series

    toplot = dios.DictOfSeries(columns=_cols)
    indexes = dict.fromkeys(_cols)

    with_data = var in data_new
    if with_data:
        dat = data_new[var]
        assert flags.index.equals(dat.index)
        toplot["data"] = dat
        nans = dat.interpolate().loc[dat.isna()]
        toplot["data-nans"] = nans

        # check for data(!) changes
        if var in data_old and not data_old[var].equals(data_new[var]):
            datold = data_old[var]
            flagsold = flagger_old.flags[var]
            assert flagsold.index.equals(datold.index)
            toplot["ref-data"] = datold
            nans = datold.interpolate().loc[datold.isna()]
            toplot["ref-data-nans"] = nans
            mask = flagger_old.isFlagged(var, flag=flagger_old.UNFLAGGED, comparator='!=')
            toplot["ref-flags"] = datold[mask]
    else:
        dummy = pd.Series(0, index=flags.index)
        toplot["data"] = dummy

    if var in flagger_old.flags:
        oldflags = flagger_old.flags[var]

        # we dont allow index-changes, because they can
        # potentially lead to wrong plotted flags
        if oldflags.index.equals(flags.index):

            old, new = _split_old_and_new(oldflags, flags)
            indexes["old-flags"] = old
            flags = flags.loc[new]

        # we will plot reference flags without data
        elif not with_data:
            dummy = pd.Series(0, index=oldflags.index)
            toplot["ref-data"] = dummy
            mask = flagger_old.isFlagged(var, flag=flagger_old.UNFLAGGED, comparator='!=')
            toplot["ref-flags"] = oldflags[mask]

    # toplot["flag-nans"] = flags[flags.isna()]
    flags = flags.dropna()

    g, s, b, u = _split_by_flag(flags, flagger_new)
    indexes["bad"] = b
    indexes["good"] = g
    indexes["suspicious"] = s
    indexes["unflagged"] = u

    # project flags to correct y-location
    for k, idx in indexes.items():
        if idx is None:
            continue
        # either actual data series or dummy
        data = toplot["data"]
        toplot[k] = data.loc[idx]

    def _plot(ax, field):
        if _colors.get(field, False):
            ax.plot(toplot[field], **_colors[field])

    # plot reference
    if not toplot["ref-data"].empty:
        fig, axs = plt.subplots(2, 1, figsize=_figsize)
        _plot(axs[0], "ref-data")
        _plot(axs[0], "ref-data-nans")
        _plot(axs[0], "ref-flags")
        ax = axs[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=_figsize)

    # plot data
    toplot["data-line"] = toplot['data']
    cols = toplot.columns.difference(["ref-data", "ref-data-nans", "ref-flags"], sort=False)
    for c in cols:
        if c not in toplot or toplot[c].empty:
            continue
        _plot(ax, c)

    plt.legend()
    plt.show()




def _split_old_and_new(old: pd.Series, new: pd.Series):
    """
    Return two indexes, that represent the equal-data and non-eq-data locations.

    Notes
    ----
        Locations that are only present in old are ignored.
        Nan's are ignored.
    """
    idx = new.dropna().index & old.dropna().index
    mask = new.loc[idx] == old[idx]
    old_idx = mask[mask].index
    new_idx = mask[~mask].index
    new_idx |= new_idx ^ new.index
    return old_idx, new_idx


def _split_by_flag(flags, flagger):
    """
    Splits flags in four separate bins, GOOD, DOUBTFUL, BAD, UNFLAGGED.

    Parameters
    ----------
    flags : pd.Series

    flagger : sqqc.Flagger

    Returns
    -------
    tuple[pd.Index]
        four indexes with locations of:
        1st GOOD's, 2nd DOUBTFUL's, 3rd BAD's, 4th UNFLAGGED
    """
    g = flags[flags == flagger.GOOD].index
    b = flags[flags == flagger.BAD].index
    d = flags[(flags > flagger.BAD) & (flags < flagger.GOOD)].index
    u = flags[flags == flagger.UNFLAGGED].index
    assert len(u) + len(g) + len(b) + len(d) <= len(flags)
    return g, d, b, u


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
            logging.warning(f"Cannot plot column '{var}', because it is not present in data.")
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
