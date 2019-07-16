#! /usr/bin/env python
# -*- coding: utf-8 -*-

from warnings import warn

def plot(data, flags, flagmask, varname, flagger, interactive_backend=True, title="Data Plot"):

    import matplotlib as mpl

    # the flagmask is True for flags to be shown False otherwise
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
        varname = set([varname])

    tmp = []
    for var in varname:
        if var not in data.columns:
            warn(f"Cannot plot column '{var}' that is not present in data.", UserWarning)
        else:
            tmp.append(var)
    if tmp:
        varname = tmp
    else:
        return

    def plot_vline(plt, points, color='blue'):
        # workaround for ax.vlines() as this work unexpected
        for point in points:
            plt.axvline(point, color=color, linestyle=':')

    def _plot(varname, ax):
        x = data.index
        y = data[varname]
        flags_ = flags[varname]
        nrofflags = len(flagger.flags.categories)
        ax.plot(x, y, '-', markersize=1, color='silver')
        if nrofflags == 3:
            colors = {0:'silver', 1:'lime', 2:'red'}
        elif nrofflags == 4:
            colors = {0:'silver', 1:'lime', 2:'yellow', 3:'red'}
        else:
            warn(f"To many flags.", UserWarning)


        # plot (all) data in silver
        ax.plot(x, y, '-', color='silver', label='data')
        # plot (all) missing data in silver
        nans = y.isna()
        flagged = flagger.isFlagged(flags_)
        idx = y.index[nans & ~flagged]
        # ax.vlines(idx, *ylim, linestyles=':', color='silver', label="missing")
        plot_vline(ax, idx, color='silver')

        # plot all flagged data in black
        ax.plot(x[flagged], y[flagged], '.', color='black', label="flagged by other test")
        # plot all flagged missing data (flagged before) in black
        idx = y.index[nans & flagged & ~flagmask]
        # ax.vlines(idx, *ylim, linestyles=':', color='black')
        plot_vline(ax, idx, color='black')
        ax.set_ylabel(varname)

        # plot currently flagged data in color of flag
        for i, f in enumerate(flagger.flags):
            if i == 0:
                continue
            flagged = flagger.isFlagged(flags_, flag=f) & flagmask
            label = f"flag: {f}" if i else 'data'
            ax.plot(x[flagged], y[flagged], '.', color=colors[i], label=label)
            idx = y.index[nans & flagged]
            # ax.vlines(idx, *ylim, linestyles=':', color=colors[i])
            plot_vline(ax, idx, color=colors[i])

    plots = len(varname)
    if plots > 1:
        fig, axes = plt.subplots(plots, 1, sharex=True)
        axes[0].set_title(title)
        for i, v in enumerate(varname):
            _plot(v, axes[i])
    else:
        fig, ax = plt.subplots()
        plt.title(title)
        _plot(varname.pop(), ax)

    plt.xlabel('time')
    # dummy plot for label `missing` see plot_vline for more info
    plt.plot([], [], ':', color='silver', label="missing data")
    plt.legend()
    plt.show()

