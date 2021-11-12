#!/usr/bin/env python
from __future__ import annotations
from typing import Dict, Union, Tuple, Callable, Sequence, Any
from typing_extensions import Literal
import functools
import numpy as np
import pandas as pd
import dios

from saqc.constants import UNFLAGGED
from saqc.core.flags import Flags, History
from saqc.lib.tools import squeezeSequence, toSequence

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Callable] = {}



class FunctionWrapper:
    def __init__(
        self,
        func: Callable,
        handles: Literal["data|flags", "index"],
        datamask: Literal["all", "field"] | None,
        multivariate: bool = False,
    ):

        if handles not in ["data|flags", "index"]:
            raise ValueError(
                f"invalid register decorator argument 'handles={repr(handles)}', "
                "only 'data|flags' or 'index' are allowed."
            )

        if datamask not in ["all", "field", None]:
            raise ValueError(
                f"invalid register decorator argument 'datamask={repr(datamask)}', "
                f"only None, 'all' or 'field' are allowed."
            )

        self.func = func
        self.func_name = func.__name__
        self.handles = handles
        self.datamask = datamask

        # set in _determineActions
        self.data_masking = None
        self.data_demasking = None
        self.flags_squeezing = None

        self._determineActions()

        # set in __call__
        self.data = None
        self.flags = None
        self.fields = None
        self.targets = None
        self.args = None
        self.kwargs = None
        self.mask_thresh = None
        self.mask = None

        # make ourself look like the wrapped function, especially the docstring
        functools.update_wrapper(self, func)
        self._multivariate = multivariate

    def _determineActions(self):
        # masking is possible, but demasking not, because the index may changed,
        # same apply for squeezing the Flags
        if self.handles == "index":
            self.flags_squeezing = False
            self.data_demasking = False

        else:  # handles == "data|flags"
            self.flags_squeezing = True
            self.data_demasking = True

        # if we have nothing to mask, we aso have nothing to unmask
        if self.datamask is None:
            self.data_masking = False
            self.data_demasking = False
        else:
            self.data_masking = True

    def __call__(
        self, data: dios.DictOfSeries, field: str, flags: Flags, *args, **kwargs
    ) -> tuple[dios.DictOfSeries, Flags]:
        """
        This wraps a call to a saqc function.

        Before the saqc function call it copies flags and maybe mask data (inplace).
        After the call it maybe squeezes modified histories and maybe reinsert the
        masked data locations.

        If the squeezing and/or the masking and/or the demasking will happen, depends
        on the decorator keywords `handles` and `datamask`. See ``_determineActions``,
        for that.
        """
        # keep this the original values
        self.data = data
        self.flags = flags
        self.fields = toSequence(field)
        self.args = args
        self.kwargs = self._checkKwargs(kwargs)

        self.targets = toSequence(kwargs.get("target", None) or [])
        self.mask_thresh = self._getMaskingThresh()

        if self.data_masking:
            columns = self._getMaskingColumns()
            data, mask = self._maskData(
                self.data, self.flags, columns, self.mask_thresh
            )
            self.data = data
            self.mask = mask

        args, kwargs = self._prepareArgs()
        data, flags = self.func(*args, **kwargs)

        if self.flags_squeezing:
            flags = self._squeezeFlags(flags)

        if self.data_demasking:
            data = self._unmaskData(data, self.mask)

        return data, flags

    def _checkKwargs(self, kwargs: dict) -> dict[str, Any]:
        if "to_mask" in kwargs and not isinstance(
            kwargs["to_mask"], (bool, float, int)
        ):
            raise TypeError(f"'to_mask' must be of type bool or float")
        return kwargs

    def _prepareArgs(self) -> tuple[tuple, dict[str, Any]]:
        """
        Prepare the args and kwargs passed to the function
        Returns
        -------
        args: tuple
            arguments to be passed to the actual call
        kwargs: dict
            keyword-arguments to be passed to the actual call
        """
        kwargs = self.kwargs.copy()
        kwargs["to_mask"] = self.mask_thresh

        # always pass a list to multivariat functions and
        # unpack single element lists for univariate functions
        if self._multivariate:
            field = self.fields
        else:
            field = squeezeSequence(self.fields)

        args = self.data, field, self.flags.copy(), *self.args
        return args, kwargs

    def _getMaskingColumns(self) -> pd.Index:
        """
        Return columns to mask, by `datamask` (decorator keyword)

        Depending on the `datamask` kw, the following s returned:
            * None : empty pd.Index
            * 'all' : all columns from data
            * 'field': single entry Index

        Returns
        -------
        columns: pd.Index
            Data columns that need to be masked.
        """
        if self.datamask is None:
            return pd.Index([])
        if self.datamask == "all":
            return self.data.columns

        # datamask == "field"
        return pd.Index(self.targets or self.fields)

    def _getMaskingThresh(self) -> float:
        """
        Generate a float threshold by the value of the `to_mask` keyword

        Returns
        -------
        threshold: float
            All data gets masked, if the flags are equal or worse than the threshold.

        Notes
        -----
        If ``to_mask`` is **not** in the kwargs, the threshold defaults to
         - ``-np.inf``
        If boolean ``to_mask`` is found in the kwargs, the threshold defaults to
         - ``-np.inf``, if ``True``
         - ``+np.inf``, if ``False``
        If a floatish ``to_mask`` is found in the kwargs, this value is taken as the threshold.
        """
        if "to_mask" not in self.kwargs:
            return UNFLAGGED

        thresh = self.kwargs["to_mask"]

        if thresh is True:  # masking ON
            thresh = UNFLAGGED

        if thresh is False:  # masking OFF
            thresh = np.inf

        thresh = float(thresh)  # handle int

        return thresh

    def _createMeta(self) -> dict:
        return {
            "func": self.func_name,
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def _squeezeFlags(self, flags: Flags) -> dios.DictOfSeries:
        """
        Generate flags from the temporary result-flags and the original flags.

        Parameters
        ----------
        flags : Flags
            The flags-frame, which is the result from a saqc-function

        Returns
        -------
        Flags
        """
        out = self.flags.copy()
        meta = self._createMeta()
        new_columns = flags.columns.difference(self.flags.columns)

        # Note: do not call _getMaskingColumns because it takes columns from data,
        # instead of from the new flags
        if self.datamask in (None, "all"):
            columns = flags.columns
        else:  # datamask == field
            columns = pd.Index(self.targets or self.fields)

        for col in columns.union(new_columns):

            if col not in out:  # ensure existence
                out.history[col] = History(index=flags.history[col].index)

            old_history = out.history[col]
            new_history = flags.history[col]

            # We only want to add new columns, that were appended during the last
            # function call. If no such columns exist, we end up with an empty
            # new_history.
            start = len(old_history.columns)
            new_history = self._sliceHistory(new_history, slice(start, None))

            # NOTE:
            # Nothing to update -> i.e. a function did not set any flags at all.
            # This has implications for function writers: early returns out of
            # functions before `flags.__getitem__` was called once, make the
            # function call invisable to the flags/history machinery and likely
            # break translation schemes such as the `PositionalTranslator`
            if new_history.empty:
                continue

            squeezed = new_history.max(raw=True)
            out.history[col] = out.history[col].append(squeezed, meta=meta)

        return out

    @staticmethod
    def _sliceHistory(history: History, sl: slice) -> History:
        history.hist = history.hist.iloc[:, sl]
        history.meta = history.meta[sl]
        return history

    @staticmethod
    def _maskData(
        data: dios.DictOfSeries, flags: Flags, columns: Sequence[str], thresh: float
    ) -> tuple[dios.DictOfSeries, dios.DictOfSeries]:
        """
        Mask data with Nans, if the flags are worse than a threshold.
            - mask only passed `columns` (preselected by `datamask`-kw from decorator)

        Returns
        -------
        masked : dios.DictOfSeries
            masked data, same dim as original
        mask : dios.DictOfSeries
            dios holding iloc-data-pairs for every column in `data`
        """
        mask = dios.DictOfSeries(columns=columns)

        # we use numpy here because it is faster
        for c in columns:
            col_mask = _isflagged(flags[c].to_numpy(), thresh)

            if col_mask.any():
                col_data = data[c].to_numpy(dtype=np.float64)

                mask[c] = pd.Series(col_data[col_mask], index=np.where(col_mask)[0])

                col_data[col_mask] = np.nan
                data[c] = col_data

        return data, mask

    @staticmethod
    def _unmaskData(
        data: dios.DictOfSeries, mask: dios.DictOfSeries
    ) -> dios.DictOfSeries:
        """
        Restore the masked data.

        Notes
        -----
        Even if this returns data, it works inplace !
        """
        # we have two options to implement this:
        #
        # =================================
        # set new data on old
        # =================================
        # col in old, in masked, in new:
        #    index differ : old <- new (replace column)
        #    else         : old <- new (set on masked: ``old[masked & new.notna()] = new``)
        # col in new only : old <- new (create column)
        # col in old only : old (delete column)
        #
        #
        # =================================
        # set old data on new (implemented)
        # =================================
        # col in old, in masked, in new :
        #    index differ : new (keep column)
        #    else         : new <- old (set on masked, ``new[masked & new.isna()] = old``)
        # col in new only : new (keep column)
        # col in old only : new (ignore, was deleted)

        # in old, in masked, in new
        columns = mask.columns.intersection(data.columns)

        for c in columns:

            # ignore
            if data[c].empty or mask[c].empty:
                continue

            # get the positions of values to unmask
            candidates = mask[c]
            # if the mask was removed during the function call, don't replace
            unmask = candidates[data[c].iloc[candidates.index].isna().to_numpy()]
            if unmask.empty:
                continue
            data[c].iloc[unmask.index] = unmask

        return data


def register(
    handles: Literal["data|flags", "index"] = "data|flags",
    datamask: Literal["all", "field"] | None = "all",
    multivariate: bool = False,
):
    def inner(func):
        wrapper = FunctionWrapper(func, handles, datamask, multivariate)
        FUNC_MAP[wrapper.func_name] = wrapper
        return wrapper

    return inner

def _isflagged(flagscol: np.ndarray | pd.Series, thresh: float) -> np.array | pd.Series:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if not isinstance(thresh, (float, int)):
        raise TypeError(f"thresh must be of type float, not {repr(type(thresh))}")

    if thresh == UNFLAGGED:
        return flagscol > UNFLAGGED

    return flagscol >= thresh

