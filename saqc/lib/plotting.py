#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import dios.dios as dios
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from saqc.flagger import BaseFlagger

import_done = False

# order is important, because
# latter may overwrite former
_cols = [
    # data
    "data",
    "data-nans",
    # flags
    "unflagged",
    "good",
    "suspicious",
    "bad",
    # special flags
    "unchanged",
    "changed",
    "flag-nans",  # currently ignored
]

nan_repr_style = dict(marker='.', fillstyle='none', ls='none', c="lightsteelblue")

_plotstyle: Dict[str, dict] = {
    # flags
    "unflagged": dict(marker='.', ls='none', c="silver", label="UNFLAGGED"),
    "good": dict(marker='.', fillstyle='none', ls='none', c="seagreen", label="GOOD"),
    "bad": dict(marker='.', fillstyle='none', ls='none', c="firebrick", label="BAD"),
    "suspicious": dict(marker='.', fillstyle='none', ls='none', c="gold", label="SUSPICIOUS"),
    "old-flags": dict(marker='.', fillstyle='none', ls='none', c="black", label="old-flags"),
    # data
    # "data": dict(marker='.', ls='none', c="silver", label="NOT FLAGGED"),
    "data": dict(c="silver", ls='-', label="data"),
    "data-nans": dict(**nan_repr_style, label="NaN"),
    # other
    # "flag-nans": nan_repr_style,
}

# _figsize = (4, 3)
_figsize = (16, 9)


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


def plotAllHook(data, flagger, show_info_table: bool = True, ):
    _plot_multiple_variables(data_old=None,
                             data_new=data,
                             flagger_old=None,
                             flagger_new=flagger,
                             targets=flagger.flags.columns,
                             show_info_table=show_info_table
                             )
    plt.tight_layout()
    plt.show()


def plotHook(
        data_old: dios.DictOfSeries,
        data_new: dios.DictOfSeries,
        flagger_old: BaseFlagger,
        flagger_new: BaseFlagger,
        sources: List[str],
        targets: List[str],
        plot_name: str,
        show_info_table: bool = True,
):
    # TODO:
    #   - sources upper     [---]  [---]  [---]
    #   - before + tab      [-oo--^--v---] |==|
    #   - current + tab     [-oo--^-ovo--] |==|
    #   if srces is None & len(targets) == 1 and with_ref -> before, curr + tabs
    #   if srces is Nne & len(tarets) > 1 or withref is False -> all targents + tabs
    #   else and len(t) > 1
    assert len(targets) > 0

    args = locals()
    if len(targets) == 1:
        _plot_single_vaiable(data_old=data_old,
                             data_new=data_new,
                             flagger_old=flagger_old,
                             flagger_new=flagger_new,
                             sources=sources,
                             targets=targets,
                             show_reference_data=True,
                             show_info_table=show_info_table,
                             plot_name=plot_name)
    else:
        _plot_multiple_variables(data_old=data_old,
                                 data_new=data_new,
                                 flagger_old=flagger_old,
                                 flagger_new=flagger_new,
                                 targets=targets,
                                 show_info_table=show_info_table)
    plt.tight_layout()
    plt.show()


def _plot_multiple_variables(
        data_old: dios.DictOfSeries,
        data_new: dios.DictOfSeries,
        flagger_old: BaseFlagger,
        flagger_new: BaseFlagger,
        targets: List[str],
        show_info_table: bool = True,
):
    tlen = len(targets)
    show_tab = show_info_table
    allaxs = []
    nfig, ncols_rest = divmod(tlen, 5)
    nfig += 1
    ncols = [4] * nfig + [ncols_rest]
    tgen = (t for t in targets)

    gs_kw = dict(width_ratios=[5, 1])
    layout = dict(
        figsize=_figsize,
        sharex=True,
        tight_layout=True,
        # constrained_layout=True,
        gridspec_kw=gs_kw
    )

    # plot 4 plots per figure
    for n in range(nfig):
        fig, axs = plt.subplots(nrows=ncols[n], ncols=2 if show_tab else 1, **layout)
        for ax in axs:
            var = next(tgen)
            tar, _ = get_data_from_var(data_old, data_new, flagger_old, flagger_new, var)
            if show_tab:
                plot_ax, tab_ax = ax
                _plot_info_table(tab_ax, tar, _plotstyle, len(tar['data']))
            else:
                plot_ax = ax
            _plot_from_dicts(plot_ax, tar, _plotstyle)
            plot_ax.set_title(str(var))
            allaxs.append(plot_ax)

    # we join all x-axis together. Surprisingly
    # this also works between different figures :D
    ax0 = allaxs[0]
    for ax in allaxs:
        ax.get_shared_x_axes().join(ax, ax0)
        ax.autoscale()


def _plot_single_vaiable(
        data_old: dios.DictOfSeries,
        data_new: dios.DictOfSeries,
        flagger_old: BaseFlagger,
        flagger_new: BaseFlagger,
        sources: List[str],
        targets: List[str],
        show_reference_data=True,
        show_info_table: bool = True,
        plot_name=""
):
    assert len(targets) == 1
    var = targets[0]
    slen = len(sources)
    show_ref = show_reference_data
    show_tab = show_info_table
    show_srces = slen > 0

    nrows = 1
    if show_ref:
        nrows += 1
    if show_srces:
        nrows += 1
        if slen > 4:
            logging.warning(f"plotting: only first 4 of {slen} sources are shown.")
            slen = 4

    fig = plt.figure(
        constrained_layout=True,
        # tight_layout=True,
        figsize=_figsize,
    )
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
            _, src = get_data_from_var(data_old, data_new, flagger_old, flagger_new, v)
            _plot_from_dicts(ax, src, _plotstyle)
            ax.set_title(f"src{i + 1}: {v}")
            allaxs.append(ax)

    curr, ref = get_data_from_var(data_old, data_new, flagger_old, flagger_new, var)

    # plot reference data (the data as it was before the test)
    if ref and show_ref:
        if show_tab:
            plot_gs, tab_gs = outer_gs[gs_count].subgridspec(ncols=2, nrows=1, width_ratios=[5, 1])
            gs_count += 1
            ax = fig.add_subplot(tab_gs)
            _plot_info_table(ax, ref, _plotstyle, len(ref['data']))
            ax = fig.add_subplot(plot_gs)
        else:
            ax = fig.add_subplot(outer_gs[gs_count])
        _plot_from_dicts(ax, ref, _plotstyle)
        ax.set_title(f"Reference data (before the test)")
        allaxs.append(ax)

    # plot data
    if show_tab:
        plot_gs, tab_gs = outer_gs[gs_count].subgridspec(ncols=2, nrows=1, width_ratios=[5, 1])
        gs_count += 1
        ax = fig.add_subplot(tab_gs)
        _plot_info_table(ax, curr, _plotstyle, len(curr['data']))
        ax = fig.add_subplot(plot_gs)
    else:
        ax = fig.add_subplot(outer_gs[gs_count])
    _plot_from_dicts(ax, curr, _plotstyle)
    ax.set_title(f"{plot_name}")
    if ref and show_ref:
        # also share y axis with ref
        ax.get_shared_y_axes().join(ax, allaxs[-1])
    allaxs.append(ax)

    # share all axes
    ax0 = allaxs[0]
    for ax in allaxs:
        ax.get_shared_x_axes().join(ax, ax0)
        ax.autoscale()

    # use all space
    outer_gs.tight_layout(fig)


def grind_space_finder(n):
    if n < 5:
        return n, 0, False
    if n < 9:
        return (n + 1) // 2, n // 2, False
    return 4, 4, True


def get_data_from_var(
        data_old: dios.DictOfSeries,
        data_new: dios.DictOfSeries,
        flagger_old: BaseFlagger,
        flagger_new: BaseFlagger,
        varname: str,
):
    __import_helper(ion=True)

    var = varname
    assert var in flagger_new.flags
    flags_new: pd.Series = flagger_new.flags[var]
    plotdict = get_plotdict(data_new, flags_new, flagger_new, var)
    ref_plotdict = None

    # prepare flags
    if flagger_old is not None and var in flagger_old.flags:
        flags_old = flagger_old.flags[var]
        ref_plotdict = get_plotdict(data_old, flags_old, flagger_old, var)

        # check flags-index changes:
        # if we want to know locations, where the flags has changed between old and new,
        # the index must match, otherwise, this could lead to wrong placed flags. Even
        # though the calculations would work.
        if flags_old.index.equals(flags_new.index):
            unchanged, changed = _split_old_and_new(flags_old, flags_new)
            unchanged, changed = project_flags_to_data([unchanged, changed], plotdict['data'])
            plotdict["unchanged"] = unchanged
            plotdict["changed"] = changed

            # calculate old-flags and update flags, like BADs,
            # to show only freshly new set values
            unflagged = plotdict["unflagged"]
            diff = unchanged.index.difference(unflagged.index)
            plotdict["old-flags"] = unchanged.loc[diff]
            for field in ["bad", "suspicious", "good"]:
                data = plotdict[field]
                isect = changed.index & data.index
                plotdict[field] = data.loc[isect]

    return plotdict, ref_plotdict


def get_plotdict(data: dios.DictOfSeries, flags: pd.Series, flagger, var):
    """
    Collect info and put them in a dict and creates dummy data if no data present.

    The collectend info include nan-data (projected to interpolated locations) and
    flag-info for BAD, SUSP., GOOD, UNFLAGGED, and flag-nans. Except the flag-nans
    all info is projected to the data-locations. E.g a BAD at the position N is
    projected to the data's x- and y- location at the very same position.

    Parameters
    ----------
    data
    flags
    flagger
    var

    Returns
    -------

    """
    pdict = dios.DictOfSeries(columns=_cols)
    pdict = data_to_pdict(pdict, data, flags, var)
    dat = pdict['data']
    pdict = flags_to_pdict(pdict, dat, flags, flagger, var)
    return pdict


def data_to_pdict(pdict, data: dios.DictOfSeries, flags: pd.Series, var):
    dat, nans = _get_data(data, flags, var)
    assert flags.index.equals(dat.index)
    pdict["data"] = dat
    pdict["data-nans"] = nans
    return pdict


def flags_to_pdict(pdict, data: pd.Series, flags: pd.Series, flagger, var):
    assert data.index.equals(flags.index)

    tup = _split_by_flag(flags, flagger, var)
    assert sum(map(len, tup)) == len(flags)

    g, s, b, u, n = project_flags_to_data(tup, data)

    pdict["good"] = g
    pdict["suspicious"] = s
    pdict["bad"] = b
    pdict["unflagged"] = u
    pdict["flag-nans"] = n
    return pdict


def project_flags_to_data(idxlist: List[pd.Series], data: pd.Series):
    res = []
    for item in idxlist:
        res.append(data.loc[item.index])
    return tuple(res)


def _get_data(data: dios.DictOfSeries, flags: pd.Series, var: str):
    if var in data:
        dat = data[var]
        nans = dat.interpolate().loc[dat.isna()]
    # create dummy data
    else:
        dat = pd.Series(0, index=flags.index)
        nans = pd.Index([])
    return dat, nans


def _plot(plotdict, ref_plotdict, styledict: Dict[str, dict], title="", info_table=True):
    """
    Create a plot with an optionally info-table.


    Each data stored in the plotdict is added to the very same axes (plot)
    with its own plot-parameters given in the styledict.
    If a key from plotdict is not present in the styledict the
    corresponding data is ignored, and will not plotted.

    If the optional ref_ref_plotdict is given a second axes (plot) with
    its data is generated. Same rules for plotdict also apply for ref_plotdict.

    For each axes a info-table is created that indicates the count of the data
    which is shown next to the plots.

    Parameters
    ----------
    plotdict : dict-like
        holds data to plot. the data must not have the same length, but the same
        type of data. - eg. a datetime x-axis cannot be in the same plot with an
        numeric axis.
    styledict : dict[str: dict]
        dict of dicts of params directly passed to plot
    title : str
        name of the whole thing
    info_table : bool, default True
        enable or disable the info-tables
    """

    gs_kw = dict(width_ratios=[5, 1])
    layout = dict(
        figsize=_figsize,
        sharey=True,
        sharex=True,
        tight_layout=True,
        # constrained_layout=True,
        gridspec_kw=gs_kw
    )

    # plot reference
    if ref_plotdict is not None:
        fig, axs = plt.subplots(2, 2, **layout)
        upper_ax, uptab_ax = axs[0]
        lower_ax, lowtab_ax = axs[1]

        uptab_ax.axis('tight')
        uptab_ax.axis('off')
        _plot_info_table(uptab_ax, ref_plotdict, styledict, len(ref_plotdict['data']))
        _plot_from_dicts(upper_ax, ref_plotdict, styledict)
    else:
        fig, axs = plt.subplots(1, 2, **layout)
        upper_ax, uptab_ax = None, None
        lower_ax, lowtab_ax = axs

    # plot current-test data
    _plot_from_dicts(lower_ax, plotdict, styledict)

    # info table for current
    lowtab_ax.axis('tight')
    lowtab_ax.axis('off')
    _plot_info_table(lowtab_ax, plotdict, styledict, len(plotdict['data']))

    # format figure layout
    if upper_ax is not None:
        upper_ax.legend()
        lower_ax.legend()
        upper_ax.set_title(f"before current test")
        lower_ax.set_title(f"current test: {title}")
        # plt.tight_layout()
    else:
        lower_ax.set_title(f"current test: {title}")
        lower_ax.legend()
        # plt.tight_layout()

    fig.subplots_adjust(hspace=0)
    plt.show()


def _plot_from_dicts(ax, plotdict, styledict):
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


def _plot_info_table(ax, plotdict, styledict, total):
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
        color = style.get('color', None) or style.get('c', 'white')
        if total == 0:
            length = percent = 0
        else:
            length = len(data)
            percent = length / total * 100
        tab.loc[len(tab), :] = [color, field, length, round(percent, 2)]

    # nested list of cell-colors
    ccs = np.full([len(tab.columns) - 1, len(tab)], fill_value='white', dtype=object)
    ccs[0] = tab['color']
    del tab['color']

    # create and format layout
    ax.axis('tight')
    ax.axis('off')
    tab_obj = ax.table(
        cellColours=ccs.transpose(),
        cellText=tab.iloc[:, :].values,
        colLabels=tab.columns[:],
        colWidths=[0.4, 0.3, 0.3],
        in_layout=True,
        bbox=[0.0, 0.1, 0.95, 0.8],
        loc='center',
    )
    tab_obj.auto_set_column_width(False)
    tab_obj.auto_set_font_size(False)
    tab_obj.set_fontsize(10)

    # color fix - use white text color if background is dark
    # sa: https://www.w3.org/TR/WCAG20/#relativeluminancedef
    thresh = 0.5
    for k, cell in tab_obj.get_celld().items():
        r, g, b, a = cell.get_facecolor()
        if 0.2126 * r + 0.7152 * g + 0.0722 * b < thresh:
            cell.set_text_props(c='white')


def _split_old_and_new(old: pd.Series, new: pd.Series):
    """
    Split new in two distinct series of equality and non-equality with old.

    Returns
    -------
        Two distinct series, one with locations, where the old and new data(!)
        are equal (including nans at same positions), the other with the rest
        of locations seen from new. This means, the rest marks locations, that
        are present(!) in new, but its data differs from old.
    """
    idx = old.index & new.index
    both_nan = old.loc[idx].isna() & new.loc[idx].isna()
    mask = (new.loc[idx] == old[idx]) | both_nan
    old_idx = mask[mask].index
    new_idx = new.index.difference(old_idx)
    return new.loc[old_idx], new.loc[new_idx]


def _split_by_flag(flags: pd.Series, flagger, var: str):
    """
    Splits flags in the five distinct bins: GOOD, SUSPICIOUS, BAD, UNFLAGGED and NaNs.
    """
    n = flags.isna()
    loc = flags.dropna().index
    g = flagger.isFlagged(field=var, loc=loc, flag=flagger.GOOD, comparator='==')
    b = flagger.isFlagged(field=var, loc=loc, flag=flagger.BAD, comparator='==')
    u = flagger.isFlagged(field=var, loc=loc, flag=flagger.UNFLAGGED, comparator='==')
    s = flagger.isFlagged(field=var, loc=loc, flag=flagger.BAD, comparator='>')
    s = flagger.isFlagged(field=var, loc=loc, flag=flagger.GOOD, comparator='<') & s
    return g[g], s[s], b[b], u[u], n[n]
