#!/usr/bin/env python
from __future__ import annotations

import inspect
import warnings
from typing import Dict, Tuple, Callable, Sequence, Any
import functools
import numpy as np
import pandas as pd
import dios

from saqc.constants import UNFLAGGED, FILTER_ALL
from saqc.core.flags import Flags, History
from saqc.lib.tools import squeezeSequence, toSequence

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Callable] = {}

_is_list_like = pd.api.types.is_list_like


class FunctionWrapper:
    def __init__(
        self,
        func: Callable,
        mask: list,
        demask: list,
        squeeze: list,
        multivariate: bool = False,
        handles_target: bool = False,
    ):
        # todo:
        #  - meta only is written with squeeze

        self.func = func
        self.func_name = func.__name__
        self.func_signature = inspect.signature(func)

        # ensure type and all elements exist in signature
        self._checkDecoratorKeywords(mask, demask, squeeze)

        self.decorator_mask = mask
        self.decorator_demask = demask
        self.decorator_squeeze = squeeze
        self.multivariate = multivariate
        self.handles_target = handles_target

        # set in __call__
        self.data = None
        self.flags = None
        self.fields = None
        self.args = None
        self.kwargs = None
        self.mask_thresh = None
        self.stored_data = None

        # make ourself look like the wrapped function, especially the docstring
        functools.update_wrapper(self, func)

    def _checkDecoratorKeywords(self, mask, demask, squeeze):
        params = self.func_signature.parameters.keys()
        for dec_arg, name in zip(
            [mask, demask, squeeze], ["mask", "demask", "squeeze"]
        ):
            typeerr = TypeError(
                f"type of decorator argument '{name}' must "
                f"be a list of strings, not {repr(type(dec_arg))}"
            )
            if not isinstance(dec_arg, list):
                raise typeerr
            for elem in dec_arg:
                if not isinstance(elem, str):
                    raise typeerr
                if elem not in params:
                    raise ValueError(
                        f"passed value {repr(elem)} in {repr(name)} is not an "
                        f"parameter in decorated function {repr(self.func_name)}"
                    )

    @staticmethod
    def _argnamesToColumns(names: list, values: dict):
        clist = []
        for name in names:
            value = values.get(name)  # eg. the value behind 'field'

            # NOTE: do not change order of the tests
            if value is None:
                pass
            elif isinstance(value, str):
                clist.append(value)
            # we ignore DataFrame, Series, DictOfSeries
            # and high order types alike
            elif hasattr(value, "columns"):
                pass
            elif _is_list_like(value) and all([isinstance(e, str) for e in value]):
                clist += value
        return pd.Index(clist)

    @staticmethod
    def _warn(missing, source):
        if len(missing) == 0:
            return
        action = source + "ed"
        obj = "flags" if source == "squeeze" else "data"
        warnings.warn(
            f"Column(s) {repr(missing)} cannot not be {action} "
            f"because they are not present in {obj}. ",
            RuntimeWarning,
        )

    def __call__(
        self, data: dios.DictOfSeries, field: str, flags: Flags, *args, **kwargs
    ) -> Tuple[dios.DictOfSeries, Flags]:
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

        self.mask_thresh = self._getMaskingThresh()

        # skip (data, field, flags)
        names = list(self.func_signature.parameters.keys())[3 : 3 + len(args)]
        all_args = {"field": field, **dict(zip(names, args)), **kwargs}

        # find columns that need masking
        columns = self._argnamesToColumns(self.decorator_mask, all_args)
        self._warn(columns.difference(self.data.columns).to_list(), source="mask")
        columns = columns.intersection(self.data.columns)

        masked, stored = self._maskData(
            data=self.data,
            flags=self.flags,
            columns=columns,
            thresh=self.mask_thresh,
        )
        self.data = masked
        self.stored_data = stored

        args, kwargs = self._prepareArgs()
        data, flags = self.func(*args, **kwargs)

        # find columns that need squeezing
        columns = self._argnamesToColumns(self.decorator_squeeze, all_args)
        self._warn(columns.difference(flags.columns).to_list(), source="squeeze")
        columns = columns.intersection(flags.columns)

        # if the function did not want to set any flags at all,
        # we assume a processing function that altered the flags
        # in an unpredictable manner or do nothing with the flags.
        # in either case we take the returned flags as the new truth.
        if columns.empty:
            result_flags = flags
        else:
            # even if this looks like a noop for columns=[],
            # it returns the old instead the new flags and
            # therefore ignores any possible processing changes
            result_flags = self._squeezeFlags(flags, columns)

        # find columns that need demasking
        columns = self._argnamesToColumns(self.decorator_demask, all_args)
        self._warn(columns.difference(data.columns).to_list(), source="demask")
        columns = columns.intersection(data.columns)

        result_data = self._unmaskData(data, self.stored_data, columns=columns)

        return result_data, result_flags

    @staticmethod
    def _checkKwargs(kwargs: dict) -> dict[str, Any]:
        if "dfilter" in kwargs and not isinstance(
            kwargs["dfilter"], (bool, float, int)
        ):
            raise TypeError(f"'dfilter' must be of type bool or float")
        return kwargs

    def _prepareArgs(self) -> Tuple[tuple, dict[str, Any]]:
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
        kwargs["dfilter"] = self.mask_thresh

        # always pass a list to multivariate functions and
        # unpack single element lists for univariate functions
        if self.multivariate:
            field = self.fields
        else:
            field = squeezeSequence(self.fields)

        args = self.data, field, self.flags.copy(), *self.args
        return args, kwargs

    def _getMaskingThresh(self) -> float:
        """
        Generate a float threshold by the value of the `dfilter` keyword

        Returns
        -------
        threshold: float
            All data gets masked, if the flags are equal or worse than the threshold.

        Notes
        -----
        If ``dfilter`` is **not** in the kwargs, the threshold defaults to `FILTER_ALL`.
        For any floatish value, it is taken as the threshold.
        """
        if "dfilter" not in self.kwargs:
            return FILTER_ALL
        return float(self.kwargs["dfilter"])  # handle int

    def _createMeta(self) -> dict:
        return {
            "func": self.func_name,
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def _squeezeFlags(self, flags: Flags, columns: pd.Index) -> Flags:
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
        out = self.flags.copy()  # the old flags
        meta = self._createMeta()
        for col in columns:

            # todo: shouldn't we fail or warn here or even have a explicit test upstream
            #  because the function should ensure consistence, especially because
            #  a empty history maybe issnt what is expected, but this happens silently
            if col not in out:  # ensure existence
                out.history[col] = History(index=flags.history[col].index)

            old_history = out.history[col]
            new_history = flags.history[col]

            # We only want to add new columns, that were appended during the last
            # function call. If no such columns exist, we end up with an empty
            # new_history.
            start = len(old_history.columns)
            new_history = self._sliceHistory(new_history, slice(start, None))

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
    ) -> Tuple[dios.DictOfSeries, dios.DictOfSeries]:
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
        data: dios.DictOfSeries, mask: dios.DictOfSeries, columns: pd.Index = None
    ) -> dios.DictOfSeries:
        """
        Restore the masked data.

        Notes
        -----
        - Even if this returns data, it works inplace !
        - `mask` is not a boolean mask, instead it holds the original values.
          The index of mask is numeric and represent the integer location
          in the original data.
        """
        if columns is None:
            columns = data.columns  # field was in old, is in mask and is in new
        columns = mask.columns.intersection(columns)

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
    mask: list[str],
    demask: list[str],
    squeeze: list[str],
    multivariate: bool = False,
    handles_target: bool = False,
):
    """
    Generalized decorator for any saqc functions.

    Before the call of the decorated function:
    - data gets masked by flags according to `dfilter`

    After the call of the decorated function:
    - data gets demasked (original data is written back)
    - flags gets squeezed (only one history column append per call)

    Parameters
    ----------
    mask : list of string
        A list of all parameter of the decorated function, that specify a column in
        data, that is read by the function and therefore should be masked by flags.

        The masking takes place before the call of the decorated function and
        temporary sets data to `NaN` at flagged locations. It is undone by ``demask``.
        The threshold of which data is considered to be flagged can be controlled
        via ``dfilter``, a parameter each function takes.

    demask : list of string
        A list of all parameter of the decorated function, that specify a column in
        data, that was masked (see ``mask``) and needs unmasking after the call.

        The unmasking replace all remaining(!) ``NaN`` by its original values from
        before the call of the decorated function.

    squeeze : list of string
        A list of all parameter of the decorated function, that specify a column in
        flags, that is written by the function.

        The squeezing combines multiple columns in the history of flags to one
        single column. This is because, multiple writes to flags, (eg. using
        ``flags[:,'a'] = 255`` twice) will result in multiple history columns,
        but should considered as a single column, because only one function call
        happened.

    multivariate : bool, default False
        If ``True``, the decorated function, process multiple data or flags
        columns at once. Therefore the decorated function must handle a list
        of columns in the parameter ``field``.

        If ``False``, the decorated function must take a single column (``str``)
        in ``field``.

    handles_target : bool, default False
        If ``True``, the decorated function, handles the target parameter by
        itself. Mandatory for multivariate functions.
    """

    def inner(func):
        wrapper = FunctionWrapper(
            func, mask, demask, squeeze, multivariate, handles_target
        )
        FUNC_MAP[wrapper.func_name] = wrapper
        return wrapper

    return inner


def flagging(**kwargs):
    """
    Default decorator for univariate flagging functions.

    Before the call of the decorated function:
    - `data[field]` gets masked by `flags[field]` according to `dfilter`
    After the call of the decorated function:
    - `data[field]` gets demasked (original data is written back)
    - `flags[field]` gets squeezed (only one history column append per call) if needed

    Notes
    -----
    For full control over masking, demasking and squeezing or to implement
    a multivariate function (multiple in- or outputs) use the `@register` decorator.

    See Also
    --------
        resister: generalization of of this function
    """
    if kwargs:
        raise ValueError("use '@register' to pass keywords")
    return register(mask=["field"], demask=["field"], squeeze=["field"])


def processing(**kwargs):
    """
    Default decorator for univariate processing functions.

    - no masking of data
    - no demasking of data
    - no squeezing of flags

    Notes
    -----
    For full control over masking, demasking and squeezing or to implement
    a multivariate function (multiple in- or outputs) use the `@register` decorator.

    See Also
    --------
        resister: generalization of of this function
    """
    if kwargs:
        raise ValueError("use '@register' to pass keywords")
    return register(mask=[], demask=[], squeeze=[])


def _isflagged(flagscol: np.ndarray | pd.Series, thresh: float) -> np.array | pd.Series:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if not isinstance(thresh, (float, int)):
        raise TypeError(f"thresh must be of type float, not {repr(type(thresh))}")

    if thresh == FILTER_ALL:
        return flagscol > UNFLAGGED

    return flagscol >= thresh
