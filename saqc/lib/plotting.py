#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import dios
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from saqc.common import *
from saqc.flagger import Flagger


def __importHelper():
    import matplotlib as mpl
    from pandas.plotting import register_matplotlib_converters

    # needed for datetime conversion
    register_matplotlib_converters()

    if not _interactive:
        # Import plot libs without interactivity, if not needed.
        # This ensures that we can produce an plot.png even if
        # tkinter is not installed. E.g. if one want to run this
        # on machines without X-Server aka. graphic interface.
        mpl.use("Agg")


# global switches - use is read-only
_interactive = True
_figsize = (16, 9)
_layout_data_to_table_ratio = [5, 1]
_show_info_table = True

# order is important, because
# latter may overwrite former
_cols = [
    # data - not mutually distinct
    "data",
    "data-nans",
    # flags - mutually distinct
    "unflagged",
    "good",
    "suspicious",
    "bad",
    "flag-nans",  # currently ignored
    # special flags - mutually distinct
    "unchanged",
    "changed",
]

_plotstyle: Dict[str, dict] = {
    # flags
    "unflagged": dict(marker=".", ls="none", c="silver", label="UNFLAGGED"),
    "good": dict(marker=".", fillstyle="none", ls="none", c="seagreen", label="GOOD"),
    "bad": dict(marker=".", fillstyle="none", ls="none", c="firebrick", label="BAD"),
    "suspicious": dict(marker=".", fillstyle="none", ls="none", c="gold", label="SUSPICIOUS"),
    "old-flags": dict(marker=".", fillstyle="none", ls="none", c="black", label="old-flags"),
    # data
    "data": dict(c="silver", ls="-", label="data"),
    "data-nans": dict(marker=".", fillstyle="none", ls="none", c="lightsteelblue", label="NaN"),
}


def _show():
    if _interactive:
        plt.show()


def plotAllHook(
    data, flagger, targets=None, show_info_table: bool = True, annotations: Optional[dios.DictOfSeries] = None,
):
    __importHelper()
    targets = flagger.columns if targets is None else targets
    _plotMultipleVariables(
        data_old=None,
        flagger_old=None,
        data_new=data,
        flagger_new=flagger,
        targets=targets,
        show_info_table=show_info_table,
        annotations=annotations,
    )
    plt.tight_layout()
    _show()


def plotHook(
    data_old: Optional[dios.DictOfSeries],
    data_new: dios.DictOfSeries,
    flagger_old: Optional[Flagger],
    flagger_new: Flagger,
    sources: List[str],
    targets: List[str],
    plot_name: str = "",
    annotations: Optional[dios.DictOfSeries] = None,
):
    assert len(targets) > 0
    __importHelper()

    args = dict(
        data_old=data_old,
        data_new=data_new,
        flagger_old=flagger_old,
        flagger_new=flagger_new,
        targets=targets,
        show_info_table=_show_info_table,
        annotations=annotations,
    )

    if len(targets) == 1:
        _plotSingleVariable(**args, sources=sources, show_reference_data=True, plot_name=plot_name)
    else:
        _plotMultipleVariables(**args)

    _show()


def _plotMultipleVariables(
    data_old: Optional[dios.DictOfSeries],
    data_new: dios.DictOfSeries,
    flagger_old: Optional[Flagger],
    flagger_new: Flagger,
    targets: List[str],
    show_info_table: bool = True,
    annotations=None,
):
    """
    Plot data and flags for a multiple target-variables.

    For each variable specified in targets a own plot is generated.
    If specified, a table with quantity information is shown on the
    right of each plot. If more than 4 vars are specified always
    four plots are combined and shown in a single window (figure).
    Nevertheless the x-axis between all figures are joint together.
    This allows to still zoom or scroll all plots simultaneously.

    Parameters
    ----------
    data_old
        data from the good old times
    data_new
        current state of data
    flagger_old
        flagger that hold flags corresponding to data_old
    flagger_new
        flagger that hold flags corresponding to data_new
    targets
        a list of strings, each indicating a column in flagger_new.flags
    show_info_table
        Show a info-table on the right of reference-data and data or not

    Returns
    -------
    None
    """
    show_tab = show_info_table
    tlen = len(targets)
    tgen = (t for t in targets)

    nfig, ncols_rest = divmod(tlen, 4)
    ncols = [4] * nfig
    if ncols_rest:
        nfig += 1
        ncols += [ncols_rest]

    gs_kw = dict(width_ratios=_layout_data_to_table_ratio)
    layout = dict(
        figsize=_figsize,
        sharex=True,
        tight_layout=True,
        squeeze=False,
        gridspec_kw=gs_kw if show_tab else {}
    )

    # plot max. 4 plots per figure
    allaxs = []
    for n in range(nfig):

        fig, axs = plt.subplots(nrows=ncols[n], ncols=2 if show_tab else 1, **layout)

        for ax in axs:
            var = next(tgen)
            tar, _ = _getDataFromVar(data_old, data_new, flagger_old, flagger_new, var)

            if show_tab:
                plot_ax, tab_ax = ax
                _plotInfoTable(tab_ax, tar, _plotstyle, len(tar["data"]))
            else:
                plot_ax = ax[0]

            _plotFromDicts(plot_ax, tar, _plotstyle)

            if annotations is not None and var in annotations:
                _annotate(plot_ax, tar, annotations[var])

            plot_ax.set_title(str(var))
            allaxs.append(plot_ax)

    # we join all x-axis together. Surprisingly
    # this also works between different figures :D
    ax0 = allaxs[0]
    for ax in allaxs:
        ax.get_shared_x_axes().join(ax, ax0)
        ax.autoscale()


def simplePlot(
    data: dios.DictOfSeries,
    flagger: Flagger,
    field: str,
    plot_name=None,
    show_info_table: bool = True,
    annotations=None,
):
    __importHelper()
    _plotSingleVariable(
        data_old=None,
        data_new=data,
        flagger_old=None,
        flagger_new=flagger,
        sources=[],
        targets=[field],
        show_reference_data=False,
        show_info_table=show_info_table,
        plot_name=plot_name or str(field),
        annotations=annotations,
    )
    _show()


def _plotSingleVariable(
    data_old: dios.DictOfSeries,
    data_new: dios.DictOfSeries,
    flagger_old: Flagger,
    flagger_new: Flagger,
    sources: List[str],
    targets: List[str],
    show_reference_data=True,
    show_info_table: bool = True,
    plot_name="current data",
    annotations=None,
):
    """
    Plot data and flags for a single target-variable.

    The resulting plot (the whole thing) can have up to 3 areas.

    - The first **optional upper area** show up to 4 sources, if given.
    - The **middle optional area** show the reference-plot, that show
      the target variable in the state before the last test was run.
      If specified, a table with quantity information is shown on the
      right.
    - The last **non-optional lower area**  shows the current data with
      its flags. If specified, a table with quantity information is shown
      on the right.

    Parameters
    ----------
    data_old
        data from the good old times
    data_new
        current state of data
    flagger_old
        flagger that hold flags corresponding to data_old
    flagger_new
        flagger that hold flags corresponding to data_new
    sources
        all sources that was used to change new to old
    targets
        a single(!) string that indicates flags in flagger_new.flags
    show_reference_data
        Show reference (aka. old) data, or not
    show_info_table
        Show a info-table on the right of reference-data and data or not
    plot_name
        The name of the data-plot

    Returns
    -------
    None

    """
    assert len(targets) == 1
    var = targets[0]
    slen = len(sources)

    curr, ref = _getDataFromVar(data_old, data_new, flagger_old, flagger_new, var)

    show_ref = show_reference_data and ref is not None
    show_tab = show_info_table
    show_srces = slen > 0

    nrows = 1
    if show_ref:
        nrows += 1
    if show_srces:
        nrows += 1
        if slen > 4:
            # possible future-fix: make own figure(s) with shared-x-axis for
            # all sources. axis can be shared between figures !
            logging.warning(f"plotting: only first 4 of {slen} sources are shown.")
            slen = 4

    fig = plt.figure(constrained_layout=True, figsize=_figsize,)
    outer_gs = fig.add_gridspec(ncols=1, nrows=nrows)
    gs_count = 0
    allaxs = []

    # plot srces
    if show_srces:
        srcs_gs_arr = outer_gs[gs_count].subgridspec(ncols=slen, nrows=1)
        gs_count += 1
        # NOTE: i implicit assume that all sources are available before the test run.
        # if this ever fails, one could use data instead of ref. but i can't imagine
        # any case, where this could happen -- bert.palm@ufz.de
        for i, gs in enumerate(srcs_gs_arr):
            ax = fig.add_subplot(gs)
            v = sources[i]
            _, src = _getDataFromVar(data_old, data_new, flagger_old, flagger_new, v)
            _plotFromDicts(ax, src, _plotstyle)
            ax.set_title(f"src{i + 1}: {v}")
            allaxs.append(ax)

    # plot reference data (the data as it was before the test)
    if ref and show_ref:
        ax = _plotDataWithTable(fig, outer_gs[gs_count], ref, show_tab=show_tab)
        ax.set_title(f"Reference data (before the test)")
        allaxs.append(ax)
        gs_count += 1

    # plot data
    ax = _plotDataWithTable(fig, outer_gs[gs_count], curr, show_tab=show_tab)
    ax.set_title(f"{plot_name}")
    # also share y-axis with ref
    if ref and show_ref:
        ax.get_shared_y_axes().join(ax, allaxs[-1])
    allaxs.append(ax)
    gs_count += 1

    if annotations is not None and var in annotations:
        _annotate(ax, curr, annotations[var])

    # share all x-axis
    ax0 = allaxs[0]
    for ax in allaxs:
        ax.get_shared_x_axes().join(ax, ax0)
        ax.autoscale()

    # use all space
    outer_gs.tight_layout(fig)


def _getDataFromVar(
    data_old: dios.DictOfSeries,
    data_new: dios.DictOfSeries,
    flagger_old: Flagger,
    flagger_new: Flagger,
    varname: str,
):
    """
    Extract flag and data information and store them in separate pd.Series.

    This is a helper that extract all relevant information from the flagger
    and data and prepare those information, so it can be plotted easily.
    This means, each information is stored in a separate pd.Series, whereby
    its index is always a subset of the `data`-series index (which is always
    be present). Also all info is projected to the y-coordinate of the data,
    so plotting all info in the same plot, will result in a data-plot with
    visible flags at the actual position.

    Hard constrains:
     0. var needs to be present in ``flagger_new.flags``
     1. iff var is present in data_xxx, then var need to
        be present in flags_xxx (``flagger_xxx.flags``)

    Conditions:
     2. if var is present in flags_new, but not in data_new, dummy-data is created
     3. if var is present in data_old, (see also 1.) reference info is generated


    Returns
    -------
    dict, {dict or None}
        Returns two dictionaries, the first holds the infos corresponding
        to the actual data and flags (from flagger_new), the second hold
        the infos from the state before the last test run. The second is
        ``None`` if condition 3. is not fulfilled.

        Each dict have the following keys, and hold pd.Series as values:

        - 'data': all data (with nan's if present) [3]
        - 'data-nans': nan's projected on locations from interpolated data
        - 'unflagged': flags that indicate unflagged [1][3]
        - 'good':  flags that indicate good's [1][3]
        - 'suspicious': flags that indicate suspicious'es [1][3]
        - 'bad': flags that indicate bad's [1][3]
        - 'flag-nans': nan's in flags [1][3]
        - 'unchanged': flags that kept unchanged during the last test [2]
        - 'changed': flags that did changed during the last test [2]

        Series marked with [1] are completely distinct to others marked with [1],
        and all [1]'s sum up to all flags, same apply for [2].
        The series marked with [3] could be empty, if the infos are not present.
        All infos are projected to the data locations.
    """
    var = varname
    assert var in flagger_new.columns
    flags_new: pd.Series = flagger_new[var]
    plotdict = _getPlotdict(data_new, flags_new, flagger_new, var)
    ref_plotdict = None

    # prepare flags
    if flagger_old is not None and var in flagger_old.columns:
        flags_old = flagger_old[var]
        ref_plotdict = _getPlotdict(data_old, flags_old, flagger_old, var)

        # check flags-index changes:
        # if we want to know locations, where the flags has changed between old and new,
        # the index must match, otherwise, this could lead to wrong placed flags. Even
        # though the calculations would work.
        if flags_old.index.equals(flags_new.index):
            unchanged, changed = _splitOldAndNew(flags_old, flags_new)
            unchanged, changed = _projectFlagsOntoData([unchanged, changed], plotdict["data"])
            plotdict["unchanged"] = unchanged
            plotdict["changed"] = changed

            # calculate old-flags and update flags, like BADs,
            # to show only freshly new set values
            unflagged = plotdict.get("unflagged", pd.Series(dtype=float))
            diff = unchanged.index.difference(unflagged.index)
            plotdict["old-flags"] = unchanged.loc[diff]
            for field in ["bad", "suspicious", "good"]:
                data = plotdict.get(field, pd.Series(dtype=float))
                isect = changed.index.intersection(data.index)
                plotdict[field] = data.loc[isect]

    return plotdict, ref_plotdict


def _getPlotdict(data: dios.DictOfSeries, flags: pd.Series, flagger, var):
    """
    Collect info and put them in a dict and creates dummy data if no data present.

    The collected info include nan-data (projected to interpolated locations) and
    flag-info for BAD, SUSP., GOOD, UNFLAGGED, and flag-nans. Except the flag-nans
    all info is projected to the data-locations. E.g a BAD at the position N is
    projected to the data's x- and y- location at the very same position.

    Parameters
    ----------
    data: dios.DictOfSeries
        holds the data. If data hold a series in `var` it is used,
        otherwise a dummy series is created from flags.

    flags: pd.Series
        hold the flags.

    flagger: saqc.Flagger
        flagger object, used for get flaginfo via ``flagger.isFlagged()``

    var: str
        identifies the data-series in ``data`` that correspond to ``flags``

    Returns
    -------
    dict
        Returns a dictionary with the following keys, that hold pd.Series as values:

        - 'data': all data (with nan's if present)
        - 'data-nans': nan's projected on locations from interpolated data
        - 'unflagged': flags that indicate unflagged [1]
        - 'good':  flags that indicate good's [1]
        - 'suspicious': flags that indicate suspicious'es [1]
        - 'bad': flags that indicate bad's [1]
        - 'flag-nans': nan's in flags [1]
        - 'unchanged': flags that kept unchanged during the last test [2]
        - 'changed': flags that did changed during the last test [2]

        Flags marked with [1] are completely distinct, and sum up to all flags,
        same apply for [2].

    """
    pdict = dios.DictOfSeries(columns=_cols)

    # fill data
    dat, nans = _getData(data, flags, var)
    assert dat.index.equals(flags.index)
    pdict["data"] = dat
    pdict["data-nans"] = nans

    # fill flags
    tup = _splitByFlag(flags, flagger, var)
    assert sum(map(len, tup)) == len(flags)
    g, s, b, u, n = _projectFlagsOntoData(list(tup), dat)
    pdict["good"] = g
    pdict["suspicious"] = s
    pdict["bad"] = b
    pdict["unflagged"] = u
    pdict["flag-nans"] = n

    return pdict


def _getData(data: dios.DictOfSeries, flags: pd.Series, var: str):
    """
    Get data from a dios or create a dummy data.

    A pd.Series is taken from `data` by `var`. If the
    data does not hold such series, a dummy series is
    created from flags, which have no y-information.
    If the series indeed was present, also the nan-location
    are extracted and projected to interpolated locations
    in data.

    Returns
    -------
    pd.Series, pd.Series
        the data-series and nan-locations
    """
    if var in data:
        dat = data[var]
        nans = dat.interpolate().loc[dat.isna()]
    else:
        # create dummy data
        dat = pd.Series(0, index=flags.index)
        nans = pd.Series([], index=pd.DatetimeIndex([]))
    return dat, nans


def _splitOldAndNew(old: pd.Series, new: pd.Series):
    """
    Split new in two distinct series of equality and non-equality with old.

    Returns
    -------
        Two distinct series, one with locations, where the old and new data(!)
        are equal (including nans at same positions), the other with the rest
        of locations seen from new. This means, the rest marks locations, that
        are present(!) in new, but its data differs from old.
    """
    idx = old.index.intersection(new.index)
    both_nan = old.loc[idx].isna() & new.loc[idx].isna()
    mask = (new.loc[idx] == old[idx]) | both_nan
    old_idx = mask[mask].index
    new_idx = new.index.difference(old_idx)
    return new.loc[old_idx], new.loc[new_idx]


def _splitByFlag(flags: pd.Series, flagger, var: str):
    """
    Splits flags in the five distinct bins: GOOD, SUSPICIOUS, BAD, UNFLAGGED and NaNs.
    """
    n = flags.isna()
    b = flags >= BAD
    g = flags < UNFLAGGED
    u = flags == UNFLAGGED
    s = (flags > UNFLAGGED) & (flags < BAD)
    return g[g], s[s], b[b], u[u], n[n]


def _projectFlagsOntoData(idxlist: List[pd.Series], data: pd.Series):
    """ Project flags to a xy-location, based on data. """
    res = []
    for item in idxlist:
        res.append(data.loc[item.index])
    return tuple(res)


def _plotDataWithTable(fig, gs, pdict, show_tab=True):
    """
    Plot multiple series from a dict and optionally create a info table

    Parameters
    ----------
    fig : matplotlib.figure
        figure object to place the plot and info-table in

    gs : matplotlib.GridSpec
        gridspec object which is devided in two subgridspec's,
        where the first will hold the plot the second the info-
        table. If `show_tab` is False, the plot is directly
        places in the given gridspec.

    pdict: dict or dict-like
        holds pd.Series with plotting-data.

    show_tab : bool, default True
        if True, show a table with quantity information of the data
        if False, no table is shown

    Returns
    -------
    matplotlib.Axes
        the axes object from the plot

    See Also
    --------
        _plotFromDicts()
        _plotInfoTable()
    """
    if show_tab:
        plot_gs, tab_gs = gs.subgridspec(ncols=2, nrows=1, width_ratios=_layout_data_to_table_ratio)
        ax = fig.add_subplot(tab_gs)
        _plotInfoTable(ax, pdict, _plotstyle, len(pdict["data"]))
        ax = fig.add_subplot(plot_gs)
    else:
        ax = fig.add_subplot(gs)
    _plotFromDicts(ax, pdict, _plotstyle)
    return ax


def _plotFromDicts(ax, plotdict, styledict):
    """
    Plot multiple data from a dict in the same plot.

    Each data stored in the plot dict is added to
    the very same axes (plot) with its own plot-
    Parameters that come from the styledict. If a
    key is not present in the styledict the
    corresponding data is ignored.

    Parameters
    ----------
    ax: matplotlib.Axes
        axes object to add the plot to

    plotdict: dict or dict-like
        holds pd.Series with plotting-data.

    styledict: dict
        holds dicts of kwargs that will passed to plot.

    Notes
    -----
     - changes the axes
     - styledict and plotdict must have same keys

    """
    for field in plotdict:
        data = plotdict[field]
        style = styledict.get(field, False)
        if style and len(data) > 0:
            ax.plot(data, **style)


def _annotate(ax, plotdict, txtseries: pd.Series):
    for x, txt in txtseries.iteritems():
        try:
            y = plotdict['data'].loc[x]
            if np.isnan(y):
                y = plotdict['data-nans'].loc[x]
        except KeyError:
            continue
        ax.annotate(txt, xy=(x, y), rotation=45)


def _plotInfoTable(ax, plotdict, styledict, total):
    """
    Make a nice table with information about the quantity of elements.

    Makes a table from data in plotdict, which indicated, how many
    elements each series in data have. The count is show as number
    and in percent from total.

    Parameters
    ----------
    ax: matplotlib.Axes
        axes object to add the table to

    plotdict: dict or dict-like
        holds pd.Series with plotting-data. only the length of the
        series is evaluated.

    styledict: dict
        holds dicts of kwargs that can passed to plot. currently only
        the `color`-kw (or just `c`) is evaluated.

    total: int/float
        total count used to calculate percentage

    Returns
    -------
        instance of matplotlib.table

    Notes
    -----
     changes the axes object

    """
    cols = ["color", "name", "[#]", "[%]"]
    tab = pd.DataFrame(columns=cols)

    # extract counts and color
    for field in plotdict:
        data = plotdict[field]
        style = styledict.get(field, {})
        color = style.get("color", None) or style.get("c", "white")
        if total == 0:
            length = percent = 0
        else:
            length = len(data)
            percent = length / total * 100
        tab.loc[len(tab), :] = [color, field, length, round(percent, 2)]

    # nested list of cell-colors
    ccs = np.full([len(tab.columns) - 1, len(tab)], fill_value="white", dtype=object)
    ccs[0] = tab["color"]
    del tab["color"]

    # disable the plot as we just
    # want to have the table
    ax.axis("tight")
    ax.axis("off")

    # create and format layout
    tab_obj = ax.table(
        cellColours=ccs.transpose(),
        cellText=tab.iloc[:, :].values,
        colLabels=tab.columns[:],
        colWidths=[0.4, 0.3, 0.3],
        in_layout=True,
        loc="center",
        # make the table a bit smaller than the plot
        bbox=[0.0, 0.1, 0.95, 0.8],
    )

    # Somehow the automatic font resizing doesen't work - the
    # font only can ahrink, not rise. There was a issue [1] in
    # matplotlib, but it is closed in favor of a new project [2].
    # Nevertheless i wasn't able to integrate it. Also it seems
    # that it also does **not** fix the problem, even though the
    # Readme promise else. See here:
    # [1] https://github.com/matplotlib/matplotlib/pull/14344
    # [2] https://github.com/swfiua/blume/
    # As a suitable workaround, we use a fixed font size.
    tab_obj.auto_set_column_width(False)
    tab_obj.auto_set_font_size(False)
    tab_obj.set_fontsize(10)

    # color fix - use white text color if background is dark
    # sa: https://www.w3.org/TR/WCAG20/#relativeluminancedef
    thresh = 0.5
    for k, cell in tab_obj.get_celld().items():
        r, g, b, a = cell.get_facecolor()
        if 0.2126 * r + 0.7152 * g + 0.0722 * b < thresh:
            cell.set_text_props(c="white")
