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

# order is important, because
# latter may overwrite former
_cols = [
    # 1st PLOT
    # reference
    "ref-data-line",
    "ref-data",
    "ref-data-nans",
    "ref-flags",

    # 2nd PLOT
    # other
    "flag-nans",  # currently ignored
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

nan_repr_style = dict(marker='.', fillstyle='none', ls='none', c="lightsteelblue", label="NaN")

# uncomment rows to disable
_plotstyle = {
    # flags
    # "unflagged": dict(marker='.', fillstyle="none", ls='none', c="silver", label="UNFLAGGED"),
    "good": dict(marker='.', fillstyle='none', ls='none', c="seagreen", label="GOOD"),
    "bad": dict(marker='.', fillstyle='none', ls='none', c="firebrick", label="BAD"),
    "suspicious": dict(marker='.', fillstyle='none', ls='none', c="gold", label="SUSPICIOUS"),
    "old-flags": dict(marker='.', fillstyle='none', ls='none', c="black", label="old-flags"),
    # data
    "data": dict(marker='.', ls='none', c="silver", label="NOT FLAGGED"),
    "data-line": dict(c="silver", ls='-', label="data"),
    "data-nans": nan_repr_style,
    # other
    "flag-nans": nan_repr_style,

    # reference
    # labels are omitted as they are the same (by c) like above
    "ref-data-line": dict(c="silver", ls='-'),
    "ref-data": dict(marker='.', ls='none', c="silver", ),
    "ref-data-nans": nan_repr_style,
    "ref-flags": dict(marker='.', ls='none', c="silver"),
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
        show_nans: bool = True,
):
    # todo:
    #   - new/changed data ?
    #       - new column -> new varname -> plot only new(+all flags)
    #       - changed data -> old != new -> plot new data(+all flags), old(no flags) as reference
    #       - index-change -> probably harmo -> plot new data(+all flags), old(no flags) as reference
    #   - else: split in old and new flags by diff (a!=b), plot data, old flags in black, other by color
    __import_helper(ion=True)

    if len(varnames) != 1:
        NotImplementedError("currently only single changed variables can be plotted")
    var = varnames[0]

    assert var in flagger_new.flags
    flags = flagger_new.flags[var]
    flags: pd.Series

    toplot = dios.DictOfSeries(columns=_cols)
    indexes = dict.fromkeys(_cols)

    # prepare data, and if it was changed during
    # the last test, also prepare reference data
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

    # no data is present, if a fresh-new
    # flags-column was assigned
    else:
        dummy = pd.Series(6, index=flags.index)
        toplot["data"] = dummy

    # prepare flags
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

    # project flags to data's y-location
    for k, idx in indexes.items():
        if idx is None:
            continue
        # data actually is data or dummy
        toplot[k] = toplot["data"].loc[idx]

    _plot(toplot, _plotstyle, plot_name)


def _plot(toplot, styledict, title=""):

    def add_data_to_plot(ax, field):
        if not toplot[field].empty and styledict.get(field, False):
            ax.plot(toplot[field], **styledict[field])

    mask = toplot.columns.str.startswith("ref-")

    # plot reference
    with_ref = not toplot["ref-data"].empty
    if with_ref:
        fig, axs = plt.subplots(3, 1, figsize=_figsize, sharey=True, sharex=True)
        upper_ax, lower_ax, tab = axs
        tab.table([["foo"] *3] * 1)
        toplot["ref-data-line"] = toplot['ref-data']
        for c in toplot.columns[mask]:
            add_data_to_plot(upper_ax, c)
    else:
        upper_ax = None
        fig, lower_ax = plt.subplots(1, 1, figsize=_figsize)

    # plot data
    toplot["data-line"] = toplot['data']
    for c in toplot.columns[~mask]:
        add_data_to_plot(lower_ax, c)

    # format figure layout
    # we have a upper and a lower ax aka. plot
    if with_ref:
        fig.subplots_adjust(hspace=0)
        upper_ax.set_title(f"{title}\n(upper: before change of index or data, lower: current data)")

    # we only have one plot, the 'lower'
    else:
        lower_ax.set_title(title)

    # fig.suptitle(title)
    fig.text(0,0, "foo \n" * 9, wrap=True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def _split_old_and_new(old: pd.Series, new: pd.Series):
    """
    Split in locations where old and new data are equal, and where not.

    Returns
    -------
        Two indexes, one with locations, where the old and new data(!) are equal
        (including nans), the other with the rest of locations seen from new.
        This means, the rest marks locations, that are present(!) in new, but the
        data differs from the old data.
    """
    idx = old.index & new.index
    both_nan = old.loc[idx].isna() & new.loc[idx].isna()
    mask = (new.loc[idx] == old[idx]) | both_nan
    old_idx = mask[mask].index
    new_idx = new.index.difference(old_idx)
    return old_idx, new_idx


def _split_by_flag(flags, flagger):
    """
    Splits flags in the four separate bins, GOOD, SUSPICIOUS, BAD and UNFLAGGED.

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
    s = flags[(flags > flagger.BAD) & (flags < flagger.GOOD)].index
    u = flags[flags == flagger.UNFLAGGED].index
    assert len(u) + len(g) + len(b) + len(s) <= len(flags)
    return g, s, b, u


