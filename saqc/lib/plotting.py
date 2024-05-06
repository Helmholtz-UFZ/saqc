#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.core import DictOfSeries, Flags
from saqc.lib.tools import toSequence

# default color cycle for flags markers (seaborn color palette)
MARKER_COL_CYCLE = [
    (0.00784313725490196, 0.24313725490196078, 1.0),
    (1.0, 0.48627450980392156, 0.0),
    (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
    (0.9098039215686274, 0.0, 0.043137254901960784),
    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    (0.6235294117647059, 0.2823529411764706, 0.0),
    (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
    (1.0, 0.7686274509803922, 0.0),
    (0.0, 0.8431372549019608, 1.0),
]

# default color cycle for plot colors (many-in-one-plots)
PLOT_COL_CYCLE = [(0, 0, 0)] + MARKER_COL_CYCLE  # itertools.cycle(MARKER_COL_CYCLE)

# default data plot configuration (color kwarg only effective for many-to-one-plots)
PLOT_KWARGS = {"alpha": 0.8, "linewidth": 1, "color": PLOT_COL_CYCLE}

# default figure configuration
FIG_KWARGS = {"figsize": (16, 9)}

# default flags markers configuration
SCATTER_KWARGS = {
    "marker": ["s", "D", "^", "o", "v"],
    "color": MARKER_COL_CYCLE,
    "alpha": 0.7,
    "zorder": 10,
    "edgecolors": "black",
    "s": 70,
}


def makeFig(
    data: DictOfSeries,
    field: list[str],
    flags: Flags,
    level: float,
    mode: Literal["subplots", "oneplot"] = "subplots",
    max_gap: str | None = None,
    history: Literal["valid", "complete"] | None | list[str] = "valid",
    xscope: slice | None = None,
    ax: mpl.axes.Axes | None = None,
    ax_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
):
    """
    Returns a figure object, containing data graph with flag marks for field.

    Parameters
    ----------
    data : {pd.DataFrame, DictOfSeries}
        data

    flags : {pd.DataFrame, DictOfSeries, saqc.flagger}
        Flags or flagger object

    field : str
        Name of the variable-to-plot

    level : str, float, default None
        Flaglevel above wich flagged values should be displayed.

    mode: Literal["subplots", "oneplot"] | str = "oneplot"
        How to process multiple variables to be plotted:
           * `"oneplot"` : plot all variables with their flags in one axis (default)
           * `"subplots"` : generate subplot grid where each axis contains one variable plot with associated flags
           * `"biplot"` : plotting first and second variable in field against each other in a scatter plot  (point cloud).

    max_gap : str, default None
        If None, all the points in the data will be connected, resulting in long linear
        lines, where continous chunks of data is missing. Nans in the data get dropped
        before plotting. If an Offset string is passed, only points that have a distance
        below `max_gap` get connected via the plotting line.


    history : {"valid", "complete", None, list of strings}, default "valid"
        Discriminate the plotted flags with respect to the tests they originate from.

        * "valid" - Only plot those flags, that do not get altered or "unflagged" by subsequent tests. Only list tests
          in the legend, that actually contributed flags to the overall resault.
        * "complete" - plot all the flags set and list all the tests ran on a variable. Suitable for debugging/tracking.
        * "clear" - clear plot from all the flagged values
        * None - just plot the resulting flags for one variable, without any historical meta information.
        * list of strings - for any string ``s`` in the list, plot the flags set by test labeled, ``s`` - if ``s`` is
          not present in the history labels, plot any flags, set by a test labeled ``s``

    xscope :
        Determine a chunk of the data to be plotted. ``xscope`` can be anything,
        that is a valid argument to the ``pandas.Series.__getitem__`` method.

    ax :
        If not ``None``, plot into the given ``matplotlib.Axes`` instance, instead of a
        newly created ``matplotlib.Figure``. This option offers a possibility to integrate
        ``SaQC`` plots into custom figure layouts.

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

    scatter_kwargs :
        Keywords to modify flags marker appearance. The markers are set via the
        `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_
        method and can have the options listed there.
        The following options are `saqc` specific:

        * ``"cycleskip"``: (int) start the cycle of shapes that are assigned any flag-type with a certain lag - defaults to ``0`` (no skip)

    plot_kwargs :
        Keywords to modify data line appearance. The markers are set via the
        `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_
        method and can have the options listed there.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        figure object.

    """
    xscope = xscope or slice(xscope)
    # data retrieval
    d = dict()
    na_mask = {}
    for f in field:
        chunk = data[f][xscope].rename(f)
        mask = chunk.isna()
        d[f] = chunk[~mask]
        na_mask[f] = mask

    flags_vals = {f: flags[f][xscope] for f in field}
    flags_hist = {f: flags.history[f].hist.loc[xscope] for f in field}
    flags_meta = {f: flags.history[f].meta for f in field}

    # set fontsize:
    plt.rcParams["font.size"] = (
        ax_kwargs.pop("fontsize", None) or plt.rcParams["font.size"]
    )

    plt.rcParams["figure.figsize"] = FIG_KWARGS["figsize"]

    # set default axis sharing behavior (share x axis over rows if not explicitly opted sharex=False):
    sharex = False
    if len(d) > 1:
        sharex = ax_kwargs.pop("sharex", True)

    if mode not in ["subplots", "oneplot"]:  # phaseplot
        if len(d) != 1:
            raise ValueError(
                f"mode {mode!r} not supported. Use one of 'subplots' or 'oneplot'"
            )
        f0 = field[0]
        flags_hist = flags_hist.copy()
        phase_index = data[mode][xscope].values
        phase_index_d = phase_index[~na_mask[f0]]
        na_mask[f0].index = phase_index
        d[f0].index = phase_index_d
        flags_vals[f0].index = phase_index
        flags_hist[f0].index = phase_index
        plot_kwargs = {**PLOT_KWARGS, **{"marker": "o", "linewidth": 0}}
        ax_kwargs = {"xlabel": mode, "ylabel": d[f0].name, **ax_kwargs}

    # insert nans between values mutually spaced > max_gap
    if max_gap:
        for f in field:
            if not d[f].empty:
                d[f] = _insertBlockingNaNs(d[f], max_gap)

    # figure composition
    if ax is None:
        nrows, ncols = ax_kwargs.pop("nrows", None), ax_kwargs.pop("ncols", None)
        if nrows is None and ncols is not None:
            nrows = int(np.ceil(len(d) / ncols))
        elif nrows is not None and ncols is None:
            ncols = int(np.ceil(len(d) / nrows))
        elif ncols is None and nrows is None:  # default:
            if len(d) <= 2:
                nrows, ncols = len(d), 1
            else:
                nrows, ncols = int(np.ceil(len(d) / 2)), 2
        if nrows * ncols < len(d):
            raise ValueError(
                f"Too many variables (got {len(d)}), to plot into subplot matrix of passed shape {nrows}x{ncols}"
            )

        if mode == "oneplot":
            fig, ax = plt.subplots(1, 1, sharex=sharex)
            ax_arr = np.empty(len(field)).astype(object)
            ax_arr[:] = ax
            ax = ax_arr

        else:  # mode == 'subplots'
            fig, ax = plt.subplots(nrows, ncols, sharex=sharex)
            if nrows * ncols == 1:
                ax = np.array(ax)
    else:  # custom ax passed
        fig, ax = ax.figure, np.array(ax)

    _plotVarWithFlags(
        ax,
        d,
        flags_vals,
        flags_hist,
        flags_meta,
        history,
        level,
        na_mask,
        plot_kwargs,
        ax_kwargs,
        scatter_kwargs,
        mode,
    )

    # readability formattin for the x-tick labels:
    fig.autofmt_xdate()
    return fig


def _instantiateAxesContext(
    plot_kwargs, scatter_kwargs, ax_kwargs, var_num, var_name, mode
):
    _scatter_mem = {}
    _scatter_kwargs = scatter_kwargs.copy()
    _ax_kwargs = ax_kwargs.copy()
    _scatter_mem = {}
    # pop shape/color cycles:
    cyclestart = _scatter_kwargs.pop("cycleskip", 0)
    marker_shape_cycle = itertools.cycle(toSequence(_scatter_kwargs.pop("marker")))
    marker_col_cycle = itertools.cycle(
        toSequence(
            _scatter_kwargs.pop(
                "color", plt.rcParams["axes.prop_cycle"].by_key()["color"]
            )
        )
    )
    # skip through cycles on to the desired start
    for k in range(0, cyclestart):
        next(_scatter_kwargs["color"])
        next(_scatter_kwargs["marker"])

    # assign variable specific labels/titles
    for axis_spec in ["xlabel", "ylabel", "title", "ylim"]:
        spec = _ax_kwargs.get(axis_spec, None)
        if isinstance(spec, list):
            _ax_kwargs[axis_spec] = spec[var_num]
        elif isinstance(spec, dict):
            _ax_kwargs[axis_spec] = spec.get(var_name, None)

    title = _ax_kwargs.get("title", "" if mode != "subplots" else None)
    _ax_kwargs["title"] = var_name if title is None else title

    return (
        _scatter_kwargs,
        _ax_kwargs,
        _scatter_mem,
        marker_col_cycle,
        marker_shape_cycle,
    )


def _configMarkers(
    flags_hist,
    flags_vals,
    flags_meta,
    var_name,
    var_dat,
    history,
    _scatter_kwargs,
    _scatter_mem,
    level,
    marker_shape_cycle,
    marker_col_cycle,
    test_i,
):
    test_i_meta = flags_meta[var_name][test_i]
    # catch empty but existing history case (flags_meta={})
    if len(test_i_meta) == 0:
        return None, _scatter_kwargs, _scatter_mem, marker_shape_cycle, marker_col_cycle
    # retrieve label information:
    label = test_i_meta["kwargs"].get("label", None) or test_i_meta["func"]
    # are only flags from certain origins to be plotted? (than "history" is a list)
    if isinstance(history, list):
        # where to get label information from
        if label not in history:
            return (
                None,
                _scatter_kwargs,
                _scatter_mem,
                marker_shape_cycle,
                marker_col_cycle,
            )

    # assign label to current marker kwarg dictionary
    _scatter_kwargs.update({"label": label})
    # retrieve flags to be plotted
    flags_i = flags_hist[var_name][test_i].astype(float)

    if history != "complete":
        # only plot those flags, that do not get altered later on:
        mask = flags_i.eq(flags_vals[var_name])
        flags_i[~mask] = np.nan

        # Skip plot, if the test did not have no effect on the all over flagging result. This avoids
        # legend overflow
        if ~(flags_i > level).any():
            return (
                None,
                _scatter_kwargs,
                _scatter_mem,
                marker_shape_cycle,
                marker_col_cycle,
            )

    # Also skip plot, if all flagged values are np.nans (to catch flag missing and masked results mainly)
    temp_i = var_dat.index.join(flags_i.index, how="inner")
    if var_dat[temp_i][flags_i[temp_i].notna()].isna().all() or (
        "flagMissing" in test_i_meta["func"]
    ):
        return None, _scatter_kwargs, _scatter_mem, marker_shape_cycle, marker_col_cycle

    # if encountering a label already associated with some marker shape/color, use that specific shape/color
    if _scatter_kwargs["label"] in _scatter_mem.keys():
        _scatter_kwargs.update(
            {
                "color": _scatter_mem[_scatter_kwargs["label"]][1],
                "marker": _scatter_mem[_scatter_kwargs["label"]][0],
            }
        )

    # if label is new, use next color/shape in the cycle
    else:
        _scatter_kwargs.update(
            {
                "color": next(marker_col_cycle),
                "marker": next(marker_shape_cycle),
            }
        )
        _scatter_mem[_scatter_kwargs["label"]] = (
            _scatter_kwargs["marker"],
            _scatter_kwargs["color"],
        )
    return flags_i, _scatter_kwargs, _scatter_mem, marker_shape_cycle, marker_col_cycle


def _instantiatePlotContext(plot_kwargs, mode, var_name, var_num, plot_col_cycle):
    _plot_kwargs = plot_kwargs.copy()
    # get current plot color from plot color cycle
    _plot_kwargs["color"] = next(plot_col_cycle)
    # assign variable specific plot appearance
    for plot_spec in ["alpha", "linewidth", "label"]:
        spec = plot_kwargs.get(plot_spec, None)
        if isinstance(spec, list):
            _plot_kwargs[plot_spec] = spec[var_num]
        elif isinstance(spec, dict):
            _plot_kwargs[plot_spec] = spec.get(var_name, None)

    if mode == "oneplot":
        _plot_kwargs["label"] = _plot_kwargs.get("label", None) or var_name
    # when plotting in subplots, plot black line and label it as 'data' (if not opted otherwise)
    else:
        _plot_kwargs["label"] = _plot_kwargs.get("label", None) or "data"
    return _plot_kwargs


def _plotVarWithFlags(
    axes,
    dat_dict,
    flags_vals,
    flags_hist,
    flags_meta,
    history,
    level,
    na_mask,
    plot_kwargs,
    ax_kwargs,
    scatter_kwargs,
    mode,
):
    # ensure array of axes reference is flat
    axes = axes.flatten()
    # zip references for the variable loop:
    loop_ref = zip(axes, dat_dict.keys(), dat_dict.values())
    # include default settings if not modified
    scatter_kwargs = {**SCATTER_KWARGS, **scatter_kwargs}
    plot_kwargs = {**PLOT_KWARGS, **plot_kwargs}
    if mode == "subplots":
        plot_kwargs["color"] = "black"
    # pop plot cycles options (will throw error when passed on) - plot always black data lines for one-dat-to-one-ax plots:
    plot_col_cycle = plot_kwargs.pop("color")
    plot_col_cycle = itertools.cycle(toSequence(plot_col_cycle))

    for var_num, (ax, var_name, var_dat) in enumerate(loop_ref):
        # every time, axis target is fresh, reinstantiate the kwarg-contexts :
        if var_num == 0 or mode == "subplots":
            (
                _scatter_kwargs,
                _ax_kwargs,
                _scatter_mem,
                marker_col_cycle,
                marker_shape_cycle,
            ) = _instantiateAxesContext(
                plot_kwargs, scatter_kwargs, ax_kwargs, var_num, var_name, mode
            )
            ax.set(**_ax_kwargs)

        _plot_kwargs = _instantiatePlotContext(
            plot_kwargs, mode, var_name, var_num, plot_col_cycle
        )
        # plot the data
        ax.plot(var_dat, **_plot_kwargs)

        # start flags plotting
        if history:  # history information is processed
            for test_i in flags_hist[var_name].columns:
                (
                    flags_i,
                    _scatter_kwargs,
                    _scatter_mem,
                    marker_shape_cycle,
                    marker_col_cycle,
                ) = _configMarkers(
                    flags_hist,
                    flags_vals,
                    flags_meta,
                    var_name,
                    var_dat,
                    history,
                    _scatter_kwargs,
                    _scatter_mem,
                    level,
                    marker_shape_cycle,
                    marker_col_cycle,
                    test_i,
                )

                if flags_i is None:
                    continue

                # plot the flags
                _plotFlags(
                    ax,
                    var_dat,
                    flags_i,
                    na_mask[var_name],
                    level,
                    _scatter_kwargs,
                )

        else:  # history is None
            _scatter_kwargs.update(
                {"color": next(marker_col_cycle), "marker": next(marker_shape_cycle)}
            )
            _plotFlags(
                ax,
                var_dat,
                flags_vals[var_name],
                na_mask[var_name],
                level,
                _scatter_kwargs,
            )

    _formatLegend(ax, dat_dict)
    for axis in axes[:-1]:
        _formatLegend(axis, dat_dict)
    return


def _formatLegend(ax, dat_dict):
    # the legend generated might contain dublucate entries, we remove those, since dubed entries are assigned all
    # the same marker color and shape:
    legend_h, legend_l = ax.get_legend_handles_labels()
    unique_idx = np.unique(legend_l, return_index=True)[1]
    leg_h = [legend_h[idx] for idx in unique_idx]
    leg_l = [legend_l[idx] for idx in unique_idx]
    # if more than one variable is plotted, list plot line and flag marker shapes in seperate
    # legends
    h_types = np.array([isinstance(h, mpl.lines.Line2D) for h in leg_h])
    if sum(h_types) > 1:
        lines_h = np.array(leg_h)[h_types]
        lines_l = np.array(leg_l)[h_types]
        flags_h = np.array(leg_h)[~h_types]
        flags_l = np.array(leg_l)[~h_types]
        ax.add_artist(
            plt.legend(
                flags_h, flags_l, loc="lower right", title="Flags", draggable=True
            )
        )
        ax.legend(lines_h, lines_l, loc="upper right", title="Data", draggable=True)
    else:
        ax.legend(leg_h, leg_l)
    return


def _plotFlags(ax, datser, flags, na_mask, level, _scatter_kwargs):
    # print(f"kwargs={_scatter_kwargs} \n variable={datser.name}")
    is_flagged = flags.astype(float) > level
    is_flagged = is_flagged[~na_mask]
    is_flagged = datser[is_flagged[is_flagged].index]
    ax.scatter(is_flagged.index, is_flagged.values, **_scatter_kwargs)


def _insertBlockingNaNs(d, max_gap):
    i = d.index
    gaps = d.reindex(
        pd.date_range(i[0].floor(max_gap), i[-1].ceil(max_gap), freq=max_gap),
        method="bfill",
        tolerance=max_gap,
    )
    gaps = gaps[gaps.isna()]
    return d.reindex(d.index.join(gaps.index, how="outer"))
