#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import itertools
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc.core.flags import Flags
from saqc.lib.tools import toSequence
from saqc.lib.types import DiosLikeT

STATSDICT = {
    "values total": lambda x, y, z: len(x),
    "invalid total (=NaN)": lambda x, y, z: x.isna().sum(),
    "invalid percentage": lambda x, y, z: round((x.isna().sum()) / len(x), 2),
    "flagged total": lambda x, y, z: (y >= z).sum(),
    "flagged percentage": lambda x, y, z: round(((y >= z).sum()) / len(x), 2),
}

PLOT_KWARGS = {"alpha": 0.8, "linewidth": 1}
FIG_KWARGS = {"figsize": (16, 9)}

_seaborn_color_palette = [
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

SCATTER_KWARGS = {
    "marker": ["s", "D", "^", "o", "v"],
    "color": _seaborn_color_palette,
    "alpha": 0.7,
    "zorder": 10,
    "edgecolors": "black",
    "s": 70,
}


def makeFig(
    data: DiosLikeT,
    field: str,
    flags: Flags,
    level: float,
    max_gap: Optional[str] = None,
    history: Union[Optional[Literal["valid", "complete"]], list] = "valid",
    xscope: Optional[slice] = None,
    phaseplot: Optional[str] = None,
    ax_kwargs: Optional[dict] = None,
):
    """
    Returns a figure object, containing data graph with flag marks for field.

    Parameters
    ----------
    data : {pd.DataFrame, dios.DictOfSeries}
        data

    flags : {pd.DataFrame, dios.DictOfSeries, saqc.flagger}
        Flags or flagger object

    field : str
        Name of the variable-to-plot

    level : str, float, default None
        Flaglevel above wich flagged values should be displayed.

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

    xscope : slice or Offset, default None
        Parameter, that determines a chunk of the data to be plotted /
        processed. `s` can be anything, that is a valid argument to the ``pandas.Series.__getitem__`` method.

    phaseplot :


    Returns
    -------
    fig : matplotlib.pyplot.figure
        figure object.

    """

    if ax_kwargs is None:
        ax_kwargs = {}
    # data retrieval
    d = data[field]
    # data slicing:
    xscope = xscope or slice(xscope)
    d = d[xscope]
    flags_vals = flags[field][xscope]
    flags_hist = flags.history[field].hist.loc[xscope]
    flags_meta = flags.history[field].meta

    # set fontsize:
    default = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = ax_kwargs.pop("fontsize", None) or default

    # set shapecycle start:
    cyclestart = ax_kwargs.pop("cycleskip", 0)

    na_mask = d.isna()
    d = d[~na_mask]
    if phaseplot:
        flags_vals = flags_vals.copy()
        flags_hist = flags_hist.copy()
        phase_index = data[phaseplot][xscope].values
        phase_index_d = phase_index[~na_mask]
        na_mask.index = phase_index
        d.index = phase_index_d
        flags_vals.index = phase_index
        flags_hist.index = phase_index
        plot_kwargs = {**PLOT_KWARGS, **{"marker": "o", "linewidth": 0}}
        ax_kwargs = {**{"xlabel": phaseplot, "ylabel": d.name}, **ax_kwargs}
    else:
        plot_kwargs = PLOT_KWARGS

    # insert nans between values mutually spaced > max_gap
    if max_gap and not d.empty:
        d = _insertBlockingNaNs(d, max_gap)

    # figure composition
    fig = mpl.pyplot.figure(constrained_layout=True, **FIG_KWARGS)
    grid = fig.add_gridspec()
    ax = fig.add_subplot(grid[0])

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
        SCATTER_KWARGS,
        cyclestart,
    )

    plt.rcParams["font.size"] = default
    return fig


def _plotVarWithFlags(
    ax,
    datser,
    flags_vals,
    flags_hist,
    flags_meta,
    history,
    level,
    na_mask,
    plot_kwargs,
    ax_kwargs,
    scatter_kwargs,
    cyclestart,
):
    scatter_kwargs = scatter_kwargs.copy()
    ax.set_title(datser.name)
    ax.plot(datser, color="black", label="data", **plot_kwargs)
    ax.set(**ax_kwargs)
    shape_cycle = scatter_kwargs.get("marker", "o")
    shape_cycle = itertools.cycle(toSequence(shape_cycle))
    color_cycle = scatter_kwargs.get(
        "color", plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    color_cycle = itertools.cycle(toSequence(color_cycle))
    for k in range(0, cyclestart):
        next(color_cycle)
        next(shape_cycle)

    if history:
        for i in flags_hist.columns:
            if isinstance(history, list):
                meta_field = "label" if "label" in flags_meta[i].keys() else "func"
                to_plot = (
                    flags_meta[i][meta_field]
                    if flags_meta[i][meta_field] in history
                    else None
                )
                if not to_plot:
                    continue
                else:
                    hist_key = "valid"
            else:
                hist_key = history
            # catch empty but existing history case (flags_meta={})
            if len(flags_meta[i]) == 0:
                continue
            label = (
                flags_meta[i]["kwargs"].get("label", None)
                or flags_meta[i]["func"].split(".")[-1]
            )
            scatter_kwargs.update({"label": label})
            flags_i = flags_hist[i].astype(float)
            if hist_key == "complete":
                scatter_kwargs.update(
                    {"color": next(color_cycle), "marker": next(shape_cycle)}
                )
                _plotFlags(ax, datser, flags_i, na_mask, level, scatter_kwargs)
            if hist_key == "valid":
                # only plot those flags, that do not get altered later on:
                mask = flags_i.eq(flags_vals)
                flags_i[~mask] = np.nan
                # Skip plot, if the test did not have no effect on the all over flagging result. This avoids
                # legend overflow
                if ~(flags_i > level).any():
                    continue

                # Also skip plot, if all flagged values are np.nans (to catch flag missing and masked results mainly)
                temp_i = datser.index.join(flags_i.index, how="inner")
                if datser[temp_i][flags_i[temp_i].notna()].isna().all() or (
                    "flagMissing" in flags_meta[i]["func"]
                ):
                    continue

                scatter_kwargs.update(
                    {"color": next(color_cycle), "marker": next(shape_cycle)}
                )
                _plotFlags(
                    ax,
                    datser,
                    flags_i,
                    na_mask,
                    level,
                    scatter_kwargs,
                )

        ax.legend()
    else:
        scatter_kwargs.update({"color": next(color_cycle), "marker": next(shape_cycle)})
        _plotFlags(ax, datser, flags_vals, na_mask, level, scatter_kwargs)


def _plotFlags(ax, datser, flags, na_mask, level, scatter_kwargs):
    is_flagged = flags.astype(float) > level
    is_flagged = is_flagged[~na_mask]
    is_flagged = datser[is_flagged[is_flagged].index]
    ax.scatter(is_flagged.index, is_flagged.values, **scatter_kwargs)


def _insertBlockingNaNs(d, max_gap):
    i = d.index
    gaps = d.reindex(
        pd.date_range(i[0].floor(max_gap), i[-1].ceil(max_gap), freq=max_gap),
        method="bfill",
        tolerance=max_gap,
    )
    gaps = gaps[gaps.isna()]
    return d.reindex(d.index.join(gaps.index, how="outer"))
