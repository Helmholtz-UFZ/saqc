#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
from typing_extensions import Literal
from saqc.lib.tools import toSequence
import pandas as pd
import numpy as np
import matplotlib as mpl
import itertools
import matplotlib.pyplot as plt
import seaborn
from saqc.constants import *
from saqc.core import Flags
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
SCATTER_KWARGS = {
    "marker": ["s", "D", "^", "o"],
    "color": seaborn.color_palette("bright"),
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
    history: Optional[Literal["valid", "complete"]] = "valid",
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


    history : {"valid", "complete", None}, default "valid"
        Discriminate the plotted flags with respect to the tests they originate from.
        * "valid" - Only plot those flags, that do not get altered or "unflagged" by subsequent tests. Only list tests
          in the legend, that actually contributed flags to the overall resault.
        * "complete" - plot all the flags set and list all the tests ran on a variable. Suitable for debugging/tracking.
        * "clear" - clear plot from all the flagged values
        * None - just plot the resulting flags for one variable, without any historical meta information.

    xscope : slice or Offset, default None
        Parameter, that determines a chunk of the data to be plotted /
        processed. `s` can be anything, that is a valid argument to the ``pandas.Series.__getitem__`` method.

    phaseplot :


    Returns
    -------
    fig : matplotlib.pyplot.figure
        figure object.

    """

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
):
    scatter_kwargs = scatter_kwargs.copy()
    ax.set_title(datser.name)
    ax.plot(datser, color="black", **plot_kwargs)
    ax.set(**ax_kwargs)
    shape_cycle = scatter_kwargs.get("marker", "o")
    shape_cycle = itertools.cycle(toSequence(shape_cycle))
    color_cycle = scatter_kwargs.get(
        "color", plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    color_cycle = itertools.cycle(toSequence(color_cycle))
    if history:
        for i in flags_hist.columns:
            # catch empty but existing history case (flags_meta={})
            if len(flags_meta[i]) == 0:
                continue
            label = (
                flags_meta[i]["kwargs"].get("label", None)
                or flags_meta[i]["func"].split(".")[-1]
            )
            scatter_kwargs.update({"label": label})
            flags_i = flags_hist[i].astype(float)
            if history == "complete":
                scatter_kwargs.update(
                    {"color": next(color_cycle), "marker": next(shape_cycle)}
                )
                _plotFlags(ax, datser, flags_i, na_mask, level, scatter_kwargs)
            if history == "valid":
                # only plot those flags, that do not get altered later on:
                mask = flags_i.eq(flags_vals)
                flags_i[~mask] = np.nan
                # Skip plot, if the test did not have no effect on the all over flagging result. This avoids
                # legend overflow
                if ~(flags_i >= level).any():
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
    is_flagged = flags.astype(float) >= level
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
