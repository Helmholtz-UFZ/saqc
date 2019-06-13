#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
from warnings import warn

from config import Fields, Params
from funcs import flagDispatch
from dsl import parseFlag


def flagWindow(flagger, flags, mask, direction='fw', window=0, **kwargs) -> pd.Series:
    fw = False
    bw = False
    f = flagger.isFlagged(flags) & mask

    if isinstance(window, int):
        x = f.rolling(window=window + 1).sum()
        if direction in ['fw', 'both']:
            fw = x.fillna(method='bfill').astype(bool)
        if direction in ['bw', 'both']:
            bw = x.shift(-window).fillna(method='bfill').astype(bool)
    else:
        # time-based windows
        if direction in ['bw', 'both']:
            raise NotImplementedError
        fw = f.rolling(window=window, closed='both').sum().astype(bool)

    fmask = bw | fw
    flags[fmask] = flagger.setFlag(flags[fmask], **kwargs)
    return flags


def flagPeriod(flagger, flags, mask=True, flag_period=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_period, **kwargs)


def flagNext(flagger, flags, mask=True, flag_values=0, **kwargs) -> pd.Series:
    return flagWindow(flagger, flags, mask, 'fw', window=flag_values, **kwargs)


def runner(meta, flagger, data, flags=None, nodata=np.nan):
    plotvars = []

    if flags is None:
        flags = pd.DataFrame(index=data.index)

    # the required meta data columns
    fields = [Fields.VARNAME, Fields.START, Fields.END, Fields.ASSIGN]

    # NOTE:
    # get to know every variable from meta
    # should go into a separate function
    for idx, configrow in meta.iterrows():
        varname, _, _, assign = configrow[fields]
        if varname not in flags and \
                (varname in data or varname not in data and assign is True):
            col_flags = flagger.initFlags(pd.DataFrame(index=data.index,
                                                       columns=[varname]))
            flags = col_flags if flags.empty else flags.join(col_flags)

    # NOTE:
    # the outer loop runs over the flag tests, the inner one over the
    # variables. Switching the loop order would complicate the
    # reference to flags from other variables within the dataset
    flag_fields = meta.columns.to_series().filter(regex=Fields.FLAGS)
    for flag_pos, flag_field in enumerate(flag_fields):

        # NOTE: just an optimization
        if meta[flag_field].dropna().empty:
            continue

        for idx, configrow in meta.iterrows():

            flag_test = configrow[flag_field]
            if pd.isnull(flag_test):
                continue

            varname, start_date, end_date, _ = configrow[fields]
            func_name, flag_params = parseFlag(flag_test)

            if varname not in data and varname not in flags:
                continue

            dchunk = data.loc[start_date:end_date]
            if dchunk.empty:
                continue

            fchunk = flags.loc[start_date:end_date]

            try:
                dchunk, ffchunk = flagDispatch(func_name,
                                               dchunk, fchunk.copy(),
                                               varname,
                                               flagger, nodata=nodata,
                                               **flag_params)
            except NameError:
                raise NameError(
                    f"function name {func_name} is not definied (variable '{varname}, 'line: {idx + 1})")

            old = flagger.isFlagged(fchunk[varname])
            new = flagger.isFlagged(ffchunk[varname])
            mask = old != new

            # flag a timespan after the condition is met
            if Params.FLAGPERIOD in flag_params:
                ffchunk[varname] = flagPeriod(flagger, ffchunk[varname], mask, **flag_params)

            # flag a certain amount of values after condition is met
            if Params.FLAGVALUES in flag_params:
                ffchunk[varname] = flagNext(flagger, ffchunk[varname], mask, **flag_params)

            if Params.FLAGPERIOD in flag_params or Params.FLAGVALUES in flag_params:
                # hack as assignments above don't preserve categorical type
                ffchunk = ffchunk.astype({
                    c: flagger.flags for c in ffchunk.columns if flagger.flag_fields[0] in c})

            if flag_params.get(Params.PLOT, False):
                plotvars.append(varname)
                plot(dchunk, ffchunk, mask, varname, flagger, title=flag_test)

            data.loc[start_date:end_date] = dchunk
            flags[start_date:end_date] = ffchunk.squeeze()

        flagger.nextTest()

    # plot all together
    if plotvars:
        plot(data, flags, True, set(plotvars), flagger)

    return data, flags


def plot(data, flags, flagmask, varname, flagger, interactive_backend=True, title="Data Plot"):
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
        ax.plot(x, y, '-',markersize=1, color='silver')
        if nrofflags == 3:
            colors = {0:'silver', 1:'lime', 2:'red'}
        if nrofflags == 4:
            colors = {0:'silver', 1:'lime', 2:'yellow', 3:'red'}

        # plot (all) data in silver
        ax.plot(x, y, '-', color='silver', label='data')
        # plot (all) missing data in silver
        nans = y.isna()
        ylim = plt.ylim()
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


def prepareMeta(meta, data):
    # NOTE: an option needed to only pass tests within a file and deduce
    #       everything else from data

    # no dates given, fall back to the available index range
    if Fields.START not in meta:
        meta = meta.assign(**{Fields.START: np.nan})
    if Fields.END not in meta:
        meta = meta.assign(**{Fields.END: np.nan})

    meta = meta.fillna(
        {Fields.END: data.index.max(),
         Fields.START: data.index.min()})

    if Fields.ASSIGN not in meta:
        meta = meta.assign(**{Fields.ASSIGN: False})

    # rows without a variables name don't help much
    meta = meta.dropna(subset=[Fields.VARNAME])

    dtype = np.datetime64 if isinstance(data.index, pd.DatetimeIndex) else int

    meta[Fields.START] = meta[Fields.START].astype(dtype)
    meta[Fields.END] = meta[Fields.END].astype(dtype)

    return meta


def readData(fname, index_col, nans):
    data = pd.read_csv(
        fname, index_col=index_col, parse_dates=True,
        na_values=nans, low_memory=False)
    data.columns = [c.split(" ")[0] for c in data.columns]
    data = data.reindex(
        pd.date_range(data.index.min(), data.index.max(), freq="10min"))
    return data


if __name__ == "__main__":

    from flagger import DmpFlagger

    datafname = "resources/data.csv"
    metafname = "resources/meta.csv"

    data = readData(datafname,
                    index_col="Date Time",
                    nans=["-9999", "-9999.0"])
    meta = prepareMeta(pd.read_csv(metafname, comment="#"), data)

    flagger = DmpFlagger()
    pdata, pflags = runner(meta, flagger, data)
