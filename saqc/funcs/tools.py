#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from typing_extensions import Literal
import numpy as np
from dios import DictOfSeries

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from saqc.constants import *
from saqc.core.register import processing
from saqc.core import register, Flags
from saqc.lib.tools import periodicMask, filterKwargs
from saqc.lib.plotting import makeFig

_MPL_DEFAULT_BACKEND = mpl.get_backend()


@register(mask=[], demask=[], squeeze=[], handles_target=True)
def copyField(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: str,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Copy data and flags to a new name (preserve flags history).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to fork (copy).
    flags : saqc.Flags
        Container to store quality flags to data.
    target: str
        Target name.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        data shape may have changed relatively to the flags input.
    flags : saqc.Flags
        The quality flags of data
        Flags shape may have changed relatively to the flags input.
    """
    if field == target:
        return data, flags

    if target in flags.columns.union(data.columns):
        if not overwrite:
            raise ValueError(f"{target}: already exist")
        data, flags = dropField(data=data, flags=flags, field=target)

    data[target] = data[field].copy()
    flags.history[target] = flags.history[field].copy()

    return data, flags


@processing()
def dropField(
    data: DictOfSeries, field: str, flags: Flags, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Drops field from the data and flags.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to drop.
    flags : saqc.Flags
        Container to store quality flags to data.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        data shape may have changed relatively to the flags input.
    flags : saqc.Flags
        The quality flags of data
        Flags shape may have changed relatively to the flags input.
    """
    del data[field]
    del flags[field]
    return data, flags


@processing()
def renameField(
    data: DictOfSeries, field: str, flags: Flags, new_name: str, **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Rename field in data and flags.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to rename.
    flags : saqc.Flags
        Container to store flags of the data.
    new_name : str
        String, field is to be replaced with.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
    """
    data[new_name] = data[field]
    flags.history[new_name] = flags.history[field]
    del data[field]
    del flags[field]
    return data, flags


@register(mask=[], demask=[], squeeze=["field"])
def maskTime(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    mode: Literal["periodic", "mask_field"],
    mask_field: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    closed: bool = True,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Realizes masking within saqc.

    Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
    values or datachunks from flagging routines. This function replaces flags with UNFLAGGED
    value, wherever values are to get masked. Furthermore, the masked values get replaced by
    np.nan, so that they dont effect calculations.

    Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:

    1. dublicate "field" in the input data (`copyField`)
    2. mask the dublicated data (this, `maskTime`)
    3. apply the tests you only want to be applied onto the masked data chunks (a saqc function)
    4. project the flags, calculated on the dublicated and masked data onto the original field data
        (`concateFlags` or `flagGeneric`)
    5. drop the dublicated data (`dropField`)

    To see an implemented example, checkout flagSeasonalRange in the saqc.functions module

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    flags : saqc.Flags
        Container to store flags of the data.
    mode : {"periodic", "mask_var"}
        The masking mode.
        - "periodic": parameters "period_start", "end" are evaluated to generate a periodical mask
        - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.
    mask_field : {None, str}, default None
        Only effective if mode == "mask_var"
        Fieldname of the column, holding the data that is to be used as mask. (must be boolean series)
        Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
        indices will be calculated and values get masked where the values of the inner join are ``True``.
    start : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `end` parameter.
        See examples section below for some examples.
    end : {None, str}, default None
        Only effective if mode == "periodic"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `end` parameter.
        See examples section below for some examples.
    closed : boolean
        Wheather or not to include the mask defining bounds to the mask.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the flags input.


    Examples
    --------
    The `period_start` and `end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `period_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:

    >>> period_start = "01T15:00:00"
    >>> end = "13T17:30:00"

    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked

    >>> period_start = "01:00"
    >>> end = "04:00"

    All the values between the first and 4th minute of every hour get masked.

    >>> period_start = "01-01T00:00:00"
    >>> end = "01-03T00:00:00"

    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:

    >>> period_start = "01-01T00:00:00"
    >>> end = "02-28T23:59:59"

    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:

    >>> period_start = "22:00:00"
    >>> end = "06:00:00"

    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    datcol_idx = data[field].index

    if mode == "periodic":
        mask = periodicMask(datcol_idx, start, end, closed)
    elif mode == "mask_field":
        idx = data[mask_field].index.intersection(datcol_idx)
        mask = data.loc[idx, mask_field]
    else:
        raise ValueError("Keyword passed as masking mode is unknown ({})!".format(mode))

    data.aloc[mask, field] = np.nan
    flags[mask, field] = UNFLAGGED
    return data, flags


@register(mask=[], demask=[], squeeze=[])
def plot(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    path: Optional[str] = None,
    max_gap: Optional[str] = None,
    history: Optional[Literal["valid", "complete", "clear"]] = "valid",
    xscope: Optional[slice] = None,
    phaseplot: Optional[str] = None,
    store_kwargs: Optional[dict] = None,
    ax_kwargs: Optional[dict] = None,
    dfilter: float = FILTER_ALL,
    **kwargs,
):
    """
    Plot data and flags or store plot to file.

    There are two modes, 'interactive' and 'store', which are determind through the
    ``save_path`` keyword. In interactive mode (default) the plot is shown at runtime
    and the program execution stops until the plot window is closed manually. In
    store mode the generated plot is stored to disk and no manually interaction is
    needed.

    Parameters
    ----------
    data : {pd.DataFrame, dios.DictOfSeries}
        data

    field : str
        Name of the variable-to-plot

    flags : {pd.DataFrame, dios.DictOfSeries, saqc.flagger}
        Flags or flagger object

    path : str, default None
        If ``None`` is passed, interactive mode is entered; plots are shown immediatly
        and a user need to close them manually before execution continues.
        If a filepath is passed instead, store-mode is entered and
        the plot is stored unter the passed location.

    max_gap : str, default None
        If None, all the points in the data will be connected, resulting in long linear
        lines, where continous chunks of data is missing. Nans in the data get dropped
        before plotting. If an offset string is passed, only points that have a distance
        below `max_gap` get connected via the plotting line.

    history : {"valid", "complete", None}, default "valid"
        Discriminate the plotted flags with respect to the tests they originate from.

        * "valid" - Only plot those flags, that do not get altered or "unflagged" by subsequent tests. Only list tests
          in the legend, that actually contributed flags to the overall resault.
        * "complete" - plot all the flags set and list all the tests ran on a variable. Suitable for debugging/tracking.
        * "clear" - clear plot from all the flagged values
        * None - just plot the resulting flags for one variable, without any historical meta information.

    xscope : slice or Offset, default None
        Parameter, that determines a chunk of the data to be plotted
        processed. `xscope` can be anything, that is a valid argument to the ``pandas.Series.__getitem__`` method.

    phaseplot : str or None, default None
        If a string is passed, plot ``field`` in the phase space it forms together with the Variable ``phaseplot``.

    store_kwargs : dict, default {}
        Keywords to be passed on to the ``matplotlib.pyplot.savefig`` method, handling
        the figure storing. To store an pickle object of the figure, use the option
        ``{'pickle': True}``, but note that all other store_kwargs are ignored then.
        Reopen with: ``pickle.load(open(savepath,'w')).show()``

    ax_kwargs : dict, default {}
        Axis keywords. Change the axis labeling defaults. Most important keywords:
        'x_label', 'y_label', 'title', 'fontsize'.

    """

    interactive = path is None
    level = kwargs.get("flag", BAD)

    if dfilter < np.inf:
        data = data.copy()
        data.loc[flags[field] >= dfilter, field] = np.nan

    if store_kwargs is None:
        store_kwargs = {}

    if ax_kwargs is None:
        ax_kwargs = {}

    if interactive:
        mpl.use(_MPL_DEFAULT_BACKEND)

    else:
        mpl.use("Agg")

    fig = makeFig(
        data=data,
        field=field,
        flags=flags,
        level=level,
        max_gap=max_gap,
        history=history,
        xscope=xscope,
        phaseplot=phaseplot,
        ax_kwargs=ax_kwargs,
    )

    if interactive:
        plt.show()

    else:
        if store_kwargs.pop("pickle", False):
            with open(path, "wb") as f:
                pickle.dump(fig, f)
        else:
            fig.savefig(path, **store_kwargs)

    return data, flags
