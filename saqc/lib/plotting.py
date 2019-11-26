#! /usr/bin/env python
# -*- coding: utf-8 -*-

from warnings import warn

from ..core.config import Params

__plotvars = []


def plotAllHook(data, flags, flagger):
    if len(__plotvars) > 1:
        _plot(data, flags, True, __plotvars, flagger)


def plotHook(data, old, new, varname, do_plot, flag_test, flagger):
    if do_plot:
        __plotvars.append(varname)
        # cannot use getFlags here, because if a flag was set (e.g. with force) the
        # flag may be the same, but any additional row (e.g. comment-field) would differ
        mask = (old[varname] == new[varname]).any(axis=1)
        _plot(data, new, mask, varname, flagger, title=flag_test)


def _plot(data, flags, flagmask, varname, flagger, interactive_backend=True, title="Data Plot", show_nans=True):

    # only import if plotting is requested by the user
    import matplotlib as mpl
    if not interactive_backend:
        # Import plot libs without interactivity, if not needed. This ensures that this can
        # produce an plot.png even if tkinter is not installed. E.g. if one want to run this
        # on machines without X-Server aka. graphic interface.
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
    from matplotlib import pyplot as plt
    # needed for datetime conversion
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if not isinstance(varname, (list, set)):
        varname = [varname]
    varname = set(varname)

    # filter out variables to which no data is associated
    tmp = []
    for var in varname:
        if var in data.columns:
            tmp.append(var)
        else:
            warn(f"Cannot plot column '{var}' that is not present in data.", UserWarning)
    if not tmp:
        return
    varname = tmp

    plots = len(varname)
    if plots > 1:
        fig, axes = plt.subplots(plots, 1, sharex=True)
        axes[0].set_title(title)
        for i, v in enumerate(varname):
            _plot_qflags(data, flags, v, flagger, flagmask, axes[i], show_nans)
    else:
        fig, ax = plt.subplots()
        plt.title(title)
        _plot_qflags(data, flags, varname.pop(), flagger, flagmask, ax, show_nans)

    plt.xlabel('time')
    # dummy plot for the label `missing` see plot_vline for more info
    plt.plot([], [], ':', color='silver', label="missing data")
    plt.legend()
    if interactive_backend:
        plt.show()


def _plot_qflags(data, flags, varname, flagger, flagmask, ax, show_nans):
    ax.set_ylabel(varname)

    x = data.index
    y = data[varname]
    ax.plot(x, y, '-', markersize=1, color='silver')

    # plot all data in silver (NaNs as vertical lines)
    ax.plot(x, y, '-', color='silver', label='data')
    flagged = flagger.isFlagged(flags, varname)
    if show_nans:
        nans = y.isna()
        idx = y.index[nans & ~flagged]
        _plot_vline(ax, idx, color='silver')

    # plot all data (and nans) that are already flagged in black
    ax.plot(x[flagged], y[flagged], '.', color='black', label="flagged by other test")
    if show_nans:
        idx = y.index[nans & flagged & ~flagmask]
        _plot_vline(ax, idx, color='black')

    # plot flags in the color corresponding to the flag
    # BAD red, GOOD green, all in between aka SUSPISIOUS in yellow
    bads = flagger.isFlagged(flags, varname, flag=flagger.BAD, comparator='==') & flagmask
    good = flagger.isFlagged(flags, varname, flag=flagger.GOOD, comparator='==') & flagmask
    susp = flagger.isFlagged(flags, varname, flag=flagger.GOOD, comparator='>') & flagmask & ~bads
    flaglist = [flagger.GOOD, flagger.BAD, 'Suspicious']
    for f, flagged in zip(flaglist, [good, bads, susp]):
        label = f"flag: {f}"
        color = _get_color(f, flagger)
        ax.plot(x[flagged], y[flagged], '.', color=color, label=label)
        if show_nans:
            idx = y.index[nans & flagged]
            _plot_vline(ax, idx, color=color)


def _plot_vline(plt, points, color='blue'):
    # workaround for ax.vlines() as this work unexpected
    # normally this should work like so:
    #   ax.vlines(idx, *ylim, linestyles=':', color='silver', label="missing")
    for point in points:
        plt.axvline(point, color=color, linestyle=':')


def _get_color(flag, flagger):
    if flag == flagger.UNFLAGGED:
        return 'silver'
    elif flag == flagger.GOOD:
        return 'green'
    elif flag == flagger.BAD:
        return 'red'
    else:
        # suspicios
        return 'yellow'


