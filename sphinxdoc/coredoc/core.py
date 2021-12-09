#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
    List,
    Sequence,
    Hashable,
    Tuple,
)
from copy import deepcopy, copy as shallowcopy

import pandas as pd
import numpy as np

from dios import DictOfSeries, to_dios
from saqc.constants import BAD

from saqc.core.modules import FunctionsMixin
from saqc.core.flags import initFlagsLike, Flags
from saqc.core.history import History
from saqc.core.register import FUNC_MAP, FunctionWrapper
from saqc.core.translation import (
    TranslationScheme,
    FloatScheme,
    SimpleScheme,
    PositionalScheme,
    DmpScheme,
)
from saqc.lib.tools import toSequence, concatDios
from saqc.lib.types import ExternalFlag, OptionalNone

# the import is needed to trigger the registration
# of the built-in (test-)functions
import saqc.funcs  # noqa

# warnings
pd.set_option("mode.chained_assignment", "warn")
np.seterr(invalid="ignore")


TRANSLATION_SCHEMES = {
    "float": FloatScheme,
    "simple": SimpleScheme,
    "dmp": DmpScheme,
    "positional": PositionalScheme,
}


class SaQC(FunctionsMixin):
    _attributes = {
        "_data",
        "_flags",
        "_scheme",
        "_attrs",
        "_called",
    }

    def __init__(
        self,
        data=None,
        flags=None,
        scheme: str | TranslationScheme = "float",
        copy: bool = True,
    ):
        self._data = self._initData(data, copy)
        self._flags = self._initFlags(flags, copy)
        self._scheme = self._initTranslationScheme(scheme)
        self._called = []
        self._attrs = {}
        self._validate(reason="init")

    def _construct(self, **attributes) -> SaQC:
        """
        Construct a new `SaQC`-Object from `self` and optionally inject
        attributes with any chechking and overhead.

        Parameters
        ----------
        **attributes: any of the `SaQC` data attributes with name and value

        Note
        ----
        For internal usage only! Setting values through `injectables` has
        the potential to mess up certain invariants of the constructed object.
        """
        out = SaQC(data=DictOfSeries(), flags=Flags(), scheme=self._scheme)
        out.attrs = self._attrs
        for k, v in attributes.items():
            if k not in self._attributes:
                raise AttributeError(f"SaQC has no attribute {repr(k)}")
            setattr(out, k, v)
        return out

    def _validate(self, reason=None):
        if not self._data.columns.equals(self._flags.columns):
            msg = "Consistency broken. data and flags have not the same columns."
            if reason:
                msg += f" This was most likely caused by: {reason}"
            raise RuntimeError(msg)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: dict[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def data_raw(self) -> DictOfSeries:
        return self._data

    @property
    def flags_raw(self) -> Flags:
        return self._flags

    @property
    def data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._data.to_df()
        data.attrs = self._attrs.copy()
        return data

    @property
    def flags(self) -> pd.DataFrame:
        data: pd.DataFrame = self._scheme.backward(self._flags, attrs=self._attrs)
        data.attrs = self._attrs.copy()
        return data

    @property
    def result(self) -> SaQCResult:
        return SaQCResult(self._data, self._flags, self._attrs, self._scheme)

    def _expandFields(
        self,
        regex: bool,
        multivariate: bool,
        field: str | Sequence[str],
        target: str | Sequence[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        check and expand `field` and `target`
        """

        if regex and target is not None:
            raise NotImplementedError(
                "explicit `target` not supported with regular expressions"
            )
        # expand regular expressions
        if regex:
            fmask = self._data.columns.str.match(field)
            fields = self._data.columns[fmask].tolist()
        else:
            fields = toSequence(field)

        targets = fields if target is None else toSequence(target)

        if multivariate:
            # wrap again to generalize the down stream loop
            fields, targets = [fields], [targets]
        else:
            if len(fields) != len(targets):
                raise ValueError(
                    "expected the same number of 'field' and 'target' values"
                )
        return fields, targets

    def _wrap(self, func: FunctionWrapper):
        """
        prepare user function input:
          - expand fields and targets
          - translate user given ``flag`` values or set the default ``BAD``
          - translate user given ``dfilter`` values or set the scheme default
          - dependeing on the workflow: initialize ``target`` variables

        Here we add the following parameters to all registered functions, regardless
        of their repsective definition:
          - ``regex``
          - ``target``

        """

        def inner(
            field: str | Sequence[str],
            *args,
            target: str | Sequence[str] = None,
            regex: bool = False,
            flag: ExternalFlag | OptionalNone = OptionalNone(),
            **kwargs,
        ) -> SaQC:

            kwargs.setdefault("dfilter", self._scheme.DFILTER_DEFAULT)

            if not isinstance(flag, OptionalNone):
                # translation schemes might want to use a flag,
                # None so we introduce a special class here
                kwargs["flag"] = self._scheme(flag)

            fields, targets = self._expandFields(
                regex=regex, multivariate=func.multivariate, field=field, target=target
            )

            out = self

            for field, target in zip(fields, targets):

                fkwargs = {
                    **kwargs,
                    "field": field,
                    "target": target,
                }

                if not func.handles_target and field != target:
                    if target in self.data.columns:
                        out = out._callFunction(
                            FUNC_MAP["dropField"], field=target, **kwargs
                        )

                    out = out._callFunction(
                        FUNC_MAP["copyField"],
                        *args,
                        **fkwargs,
                    )
                    fkwargs["field"] = fkwargs.pop("target")

                out = out._callFunction(
                    func,
                    *args,
                    **fkwargs,
                )
            return out

        return inner

    def _callFunction(
        self,
        function: Callable,
        field: str | Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> SaQC:

        res = function(data=self._data, flags=self._flags, field=field, *args, **kwargs)

        # keep consistence: if we modify data and flags inplace in a function,
        # but data is the original and flags is a copy (as currently implemented),
        # data and flags of the original saqc obj may change inconsistently.
        self._data, self._flags = res
        self._called += [(field, (function, args, kwargs))]
        self._validate(reason=f"call to {repr(function.__name__)}")

        return self._construct(
            _data=self._data, _flags=self._flags, _called=self._called
        )

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to __getattr__.
        We use this mechanism to make the registered functions appear
        as `SaQC`-methods without actually implementing them.
        """
        if key not in FUNC_MAP:
            raise AttributeError(f"SaQC has no attribute {repr(key)}")
        return self._wrap(FUNC_MAP[key])

    def copy(self, deep=True):
        copyfunc = deepcopy if deep else shallowcopy
        new = self._construct()
        for attr in self._attributes:
            setattr(new, attr, copyfunc(getattr(self, attr)))
        return new

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memodict=None):
        return self.copy(deep=True)

    def _initTranslationScheme(
        self, scheme: str | TranslationScheme
    ) -> TranslationScheme:
        if isinstance(scheme, str) and scheme in TRANSLATION_SCHEMES:
            return TRANSLATION_SCHEMES[scheme]()
        if isinstance(scheme, TranslationScheme):
            return scheme
        raise TypeError(
            f"expected one of the following translation schemes '{TRANSLATION_SCHEMES.keys()} "
            f"or an initialized Translator object, got '{scheme}'"
        )

    def _initData(self, data, copy: bool) -> DictOfSeries:

        if data is None:
            return DictOfSeries()

        if isinstance(data, list):
            results = []
            for d in data:
                results.append(self._castToDios(d, copy=copy))
            return concatDios(results, warn=True, stacklevel=3)

        if isinstance(data, (DictOfSeries, pd.DataFrame, pd.Series)):
            return self._castToDios(data, copy)

        raise TypeError(
            "'data' must be of type pandas.Series, "
            "pandas.DataFrame or dios.DictOfSeries or"
            "a list of those."
        )

    def _castToDios(self, data, copy: bool):
        if isinstance(data, pd.Series):
            if not isinstance(data.name, str):
                raise ValueError(f"Cannot init from unnamed pd.Series")
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            for idx in [data.index, data.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'data' should not have MultiIndex")
        data = to_dios(data)  # noop for DictOfSeries
        for c in data.columns:
            if not isinstance(c, str):
                raise TypeError("columns labels must be of type string")
        if copy:
            data = data.copy()
        return data

    def _initFlags(self, flags, copy: bool) -> Flags:
        if flags is None:
            return initFlagsLike(self._data)

        if isinstance(flags, list):
            result = Flags()
            for f in flags:
                f = self._castToFlags(f, copy=copy)
                for c in f.columns:
                    if c in result.columns:
                        warnings.warn(
                            f"Column {c} already exist. Data is overwritten. "
                            f"Avoid duplicate columns names over all inputs.",
                            stacklevel=2,
                        )
                        result.history[c] = f.history[c]
            flags = result

        elif isinstance(flags, (pd.DataFrame, DictOfSeries, Flags)):
            flags = self._castToFlags(flags, copy=copy)

        else:
            raise TypeError(
                "'flags' must be of type pandas.DataFrame, "
                "dios.DictOfSeries or saqc.Flags or "
                "a list of those."
            )

        # sanitize
        # - if column is missing flags but present in data, add it
        # - if column is present in both, the index must be equal
        for c in self._data.columns:
            if c not in flags.columns:
                flags.history[c] = History(self._data[c].index)
            else:
                if not flags[c].index.equals(self._data[c].index):
                    raise ValueError(
                        f"The flags index of column {c} does not equals "
                        f"the index of the same column in data."
                    )
        return flags

    def _castToFlags(self, flags, copy):
        if isinstance(flags, pd.DataFrame):
            for idx in [flags.index, flags.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'flags' should not have MultiIndex")
        if not isinstance(flags, Flags):
            flags = Flags(flags)
        if copy:
            flags = flags.copy()
        return flags

    def flagMissing(self, field, flag):
        """
        Flag NaNs in data.

        By default only NaNs are flagged, that not already have a flag.
        `to_mask` can be used to pass a flag that is used as threshold.
        Each flag worse than the threshold is replaced by the function.
        This is, because the data gets masked (with NaNs) before the
        function evaluates the NaNs.

        Parameters
        ----------
        field : str
            Column(s) in flags and data.

        flag : float, default BAD
            Flag to set.
        """
        pass

    def flagIsolated(self, field, gap_window, group_window, flag):
        """
        Find and flag temporal isolated groups of data.

        The function flags arbitrarily large groups of values, if they are surrounded by
        sufficiently large data gaps. A gap is a timespan containing either no data at all
        or NaNs only.

        Parameters
        ----------
        field : str
            Column(s) in flags and data.

        gap_window : str
            Minimum gap size required before and after a data group to consider it
            isolated. See condition (2) and (3)

        group_window : str
            Maximum size of a data chunk to consider it a candidate for an isolated group.
            Data chunks that are bigger than the ``group_window`` are ignored.
            This does not include the possible gaps surrounding it.
            See condition (1).

        flag : float, default BAD
            Flag to set.

        Notes
        -----
        A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated
        timestamps :math:`t_k,t_{k+1},...,t_{k+n}`, is considered to be isolated, if:

        1. :math:`t_{k+1} - t_n <` `group_window`
        2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`,
            is valid (preceeding gap).
        3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`,
            is valid (succeding gap).
        """
        pass

    def flagJumps(self, field, thresh, window, min_periods, flag):
        """
        Flag jumps and drops in data.

        Flag data where the mean of its values significantly changes (the data "jumps").

        Parameters
        ----------
        field : str
            Column(s) in flags and data.

        thresh : float
            Threshold value by which the mean of data has to change to trigger flagging.

        window : str
            Size of the moving window. This is the number of observations used
            for calculating the statistic.

        min_periods : int, default 1
            Minimum number of observations in window required to calculate a valid
            mean value.

        flag : float, default BAD
            Flag to set.
        """
        pass

    def flagChangePoints(
        self,
        field,
        stat_func,
        thresh_func,
        window,
        min_periods,
        closed,
        reduce_window,
        reduce_func,
        flag,
    ):
        """
        Flag data where it significantly changes.

        Flag data points, where the parametrization of the process, the data is assumed to
        generate by, significantly changes.

        The change points detection is based on a sliding window search.

        Parameters
        ----------
        field : str
            A column in flags and data.

        stat_func : Callable
             A function that assigns a value to every twin window. The backward-facing
             window content will be passed as the first array, the forward-facing window
             content as the second.

        thresh_func : Callable
            A function that determines the value level, exceeding wich qualifies a
            timestamps func value as denoting a change-point.

        window : str, tuple of str
            Size of the moving windows. This is the number of observations used for
            calculating the statistic.

            If it is a single frequency offset, it applies for the backward- and the
            forward-facing window.

            If two offsets (as a tuple) is passed the first defines the size of the
            backward facing window, the second the size of the forward facing window.

        min_periods : int or tuple of int
            Minimum number of observations in a window required to perform the changepoint
            test. If it is a tuple of two int, the first refer to the backward-,
            the second to the forward-facing window.

        closed : {'right', 'left', 'both', 'neither'}, default 'both'
            Determines the closure of the sliding windows.

        reduce_window : str or None, default None
            The sliding window search method is not an exact CP search method and usually
            there wont be detected a single changepoint, but a "region" of change around
            a changepoint.

            If `reduce_window` is given, for every window of size `reduce_window`, there
            will be selected the value with index `reduce_func(x, y)` and the others will
            be dropped.

            If `reduce_window` is None, the reduction window size equals the twin window
            size, the changepoints have been detected with.

        reduce_func : Callable, default ``lambda x, y: x.argmax()``
            A function that must return an index value upon input of two arrays x and y.
            First input parameter will hold the result from the stat_func evaluation for
            every reduction window. Second input parameter holds the result from the
            `thresh_func` evaluation.
            The default reduction function just selects the value that maximizes the
            `stat_func`.

        flag : float, default BAD
            flag to set.
        """
        pass

    def assignChangePointCluster(
        self,
        field,
        stat_func,
        thresh_func,
        window,
        min_periods,
        closed,
        reduce_window,
        reduce_func,
        model_by_resids,
    ):
        """
        Label data where it changes significantly.

        The labels will be stored in data. Unless `target` is given the labels will
        overwrite the data in `field`. The flags will always set to `UNFLAGGED`.

        Assigns label to the data, aiming to reflect continuous regimes of the processes
        the data is assumed to be generated by. The regime change points detection is
        based on a sliding window search.


        Parameters
        ----------
        field : str
            The reference variable, the deviation from wich determines the flagging.

        stat_func : Callable[[numpy.array, numpy.array], float]
            A function that assigns a value to every twin window. Left window content will
            be passed to first variable,
            right window content will be passed to the second.

        thresh_func : Callable[numpy.array, numpy.array], float]
            A function that determines the value level, exceeding wich qualifies a
            timestamps func func value as denoting a changepoint.

        window : str, tuple of string
            Size of the rolling windows the calculation is performed in. If it is a single
            frequency offset, it applies for the backward- and the forward-facing window.

            If two offsets (as a tuple) is passed the first defines the size of the
            backward facing window, the second the size of the forward facing window.

        min_periods : int or tuple of int
            Minimum number of observations in a window required to perform the changepoint
            test. If it is a tuple of two int, the first refer to the backward-,
            the second to the forward-facing window.

        closed : {'right', 'left', 'both', 'neither'}, default 'both'
            Determines the closure of the sliding windows.

        reduce_window : {None, str}, default None
            The sliding window search method is not an exact CP search method and usually
            there wont be detected a single changepoint, but a "region" of change around
            a changepoint. If `reduce_window` is given, for every window of size
            `reduce_window`, there will be selected the value with index `reduce_func(x,
            y)` and the others will be dropped. If `reduce_window` is None, the reduction
            window size equals the twin window size, the changepoints have been detected
            with.

        reduce_func : callable, default lambda x,y: x.argmax()
            A function that must return an index value upon input of two arrays x and y.
            First input parameter will hold the result from the stat_func evaluation for
            every reduction window. Second input parameter holds the result from the
            thresh_func evaluation. The default reduction function just selects the value
            that maximizes the stat_func.

        model_by_resids : bool, default False
            If True, the results of `stat_funcs` are written, otherwise the regime labels.
        """
        pass

    def flagConstants(self, field, thresh, window, flag):
        """
        Flag constant data values.

        Flags plateaus of constant data if their maximum total change in
        a rolling window does not exceed a certain threshold.

        Any interval of values y(t),...,y(t+n) is flagged, if:
         - (1): n > ``window``
         - (2): abs(y(t + i) - (t + j)) < `thresh`, for all i,j in [0, 1, ..., n]

        Parameters
        ----------
        field : str
            A column in flags and data.

        thresh : float
            Maximum total change allowed per window.

        window : str | int
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window.
            Each window will be a variable sized based on the observations included
            in the time-period.

        flag : float, default BAD
            Flag to set.
        """
        pass

    def flagByVariance(self, field, window, thresh, maxna, maxna_group, flag):
        """
        Flag low-variance data.

        Flags plateaus of constant data if the variance in a rolling window does not
        exceed a certain threshold.

        Any interval of values y(t),..y(t+n) is flagged, if:

        (1) n > `window`
        (2) variance(y(t),...,y(t+n) < `thresh`

        Parameters
        ----------
        field : str
            A column in flags and data.

        window : str | int
            Size of the moving window. This is the number of observations used
            for calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window.
            Each window will be sized, based on the number of observations included
            in the time-period.

        thresh : float, default 0.0005
            Maximum total variance allowed per window.

        maxna : int, default None
            Maximum number of NaNs allowed in window.
            If more NaNs are present, the window is not flagged.

        maxna_group : int, default None
            Same as `maxna` but for consecutive NaNs.

        flag : float, default BAD
            Flag to set.
        """
        pass

    def fitPolynomial(self, field, window, order, min_periods):
        """
        Fits a polynomial model to the data.

        The fit is calculated by fitting a polynomial of degree `order` to a data slice
        of size `window`, that has x at its center.

        Note that the result is stored in `field` and overwrite it unless a
        `target` is given.

        In case your data is sampled at an equidistant frequency grid:

        (1) If you know your data to have no significant number of missing values,
        or if you do not want to calculate residues for windows containing missing values
        any way, performance can be increased by setting min_periods=window.

        Note, that the initial and final window/2 values do not get fitted.

        Each residual gets assigned the worst flag present in the interval of
        the original data.

        Parameters
        ----------
        field : str
            A column in flags and data.

        window : str, int
            Size of the window you want to use for fitting. If an integer is passed,
            the size refers to the number of periods for every fitting window. If an
            offset string is passed, the size refers to the total temporal extension. The
            window will be centered around the vaule-to-be-fitted. For regularly sampled
            data always a odd number of periods will be used for the fit (periods-1 if
            periods is even).

        order : int
            Degree of the polynomial used for fitting

        min_periods : int or None, default 0
            Minimum number of observations in a window required to perform the fit,
            otherwise NaNs will be assigned.
            If ``None``, `min_periods` defaults to 1 for integer windows and to the
            size of the window for offset based windows.
            Passing 0, disables the feature and will result in over-fitting for too
            sparse windows.
        """
        pass

    def flagDriftFromNorm(self, field, freq, spread, frac, metric, method, flag):
        """
        Flags data that deviates from an avarage data course.

        "Normality" is determined in terms of a maximum spreading distance,
        that members of a normal group must not exceed. In addition, only a group is considered
        "normal" if it contains more then `frac` percent of the variables in "field".

        See the Notes section for a more detailed presentation of the algorithm

        Parameters
        ----------
        field : str
            A column in flags and data.

        freq : str
            Frequency, that split the data in chunks.

        spread : float
            Maximum spread allowed in the group of *normal* data. See Notes section for more details.

        frac : float, default 0.5
            Fraction defining the normal group. Use a value from the interval [0,1].
            The higher the value, the more stable the algorithm will be. For values below
            0.5 the results are undefined.

        metric : Callable, default ``lambda x,y:pdist(np.array([x,y]),metric="cityblock")/len(x)``
            Distance function that takes two arrays as input and returns a scalar float.
            This value is interpreted as the distance of the two input arrays.
            Defaults to the `averaged manhattan metric` (see Notes).

        method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
            Linkage method used for hierarchical (agglomerative) clustering of the data.
            `method` is directly passed to ``scipy.hierarchy.linkage``. See its documentation [1] for
            more details. For a general introduction on hierarchical clustering see [2].

        flag : float, default BAD
            flag to set.

        Notes
        -----
        following steps are performed for every data "segment" of length `freq` in order to find the
        "abnormal" data:

        1. Calculate distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `field`.
           (with :math:`d` denoting the distance function, specified by `metric`.
        2. Calculate a dendogram with a hierarchical linkage algorithm, specified by `method`.
        3. Flatten the dendogram at the level, the agglomeration costs exceed `spread`
        4. check if a cluster containing more than `frac` variables.

            1. if yes: flag all the variables that are not in that cluster (inside the segment)
            2. if no: flag nothing

        The main parameter giving control over the algorithms behavior is the `spread` parameter,
        that determines the maximum spread of a normal group by limiting the costs, a cluster
        agglomeration must not exceed in every linkage step.
        For singleton clusters, that costs just equal half the distance, the data in the
        clusters, have to each other. So, no data can be clustered together, that are more then
        2*`spread` distances away from each other. When data get clustered together, this new
        clusters distance to all the other data/clusters is calculated according to the linkage
        method specified by `method`. By default, it is the minimum distance, the members of the
        clusters have to each other. Having that in mind, it is advisable to choose a distance
        function, that can be well interpreted in the units dimension of the measurement and where
        the interpretation is invariant over the length of the data. That is, why,
        the "averaged manhattan metric" is set as the metric default, since it corresponds to the
        averaged value distance, two data sets have (as opposed by euclidean, for example).

        References
        ----------
        Documentation of the underlying hierarchical clustering algorithm:
            [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        Introduction to Hierarchical clustering:
            [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
        """
        pass

    def flagDriftFromReference(
        self, field, freq, reference, thresh, metric, target, flag
    ):
        """
        Flags data that deviates from a reference course.

        The deviation is measured by a passed distance function.

        Parameters
        ----------
        field : str
            A column in flags and data.

        freq : str
            Frequency, that split the data in chunks.

        reference : str
            Reference variable, the deviation is calculated from.

        thresh : float
            Maximum deviation from reference.

        metric : Callable
            Distance function. Takes two arrays as input and returns a scalar float.
            This value is interpreted as the mutual distance of the two input arrays.
            Defaults to the `averaged manhattan metric` (see Notes).

        target : None
            Ignored.

        flag : float, default BAD
            Flag to set.

        Notes
        -----
        It is advisable to choose a distance function, that can be well interpreted in
        the units dimension of the measurement and where the interpretation is invariant over the
        length of the data. That is, why, the "averaged manhatten metric" is set as the metric
        default, since it corresponds to the averaged value distance, two data sets have (as opposed
        by euclidean, for example).
        """
        pass

    def correctDrift(self, field, maintenance_field, model, cal_range):
        """
        The function corrects drifting behavior.

        See the Notes section for an overview over the correction algorithm.

        Parameters
        ----------
        field : str
            Column in data and flags.

        maintenance_field : str
            Column holding the support-points information.
            The data is expected to have the following form:
            The index of the series represents the beginning of a maintenance
            event, wheras the values represent its endings.

        model : Callable or {'exponential', 'linear'}
            A modelfunction describing the drift behavior, that is to be corrected.
            Either use built-in exponential or linear drift model by passing a string, or pass a custom callable.
            The model function must always contain the keyword parameters 'origin' and 'target'.
            The starting parameter must always be the parameter, by wich the data is passed to the model.
            After the data parameter, there can occure an arbitrary number of model calibration arguments in
            the signature.
            See the Notes section for an extensive description.

        cal_range : int, default 5
            Number of values to calculate the mean of, for obtaining the value level directly
            after and directly before a maintenance event. Needed for shift calibration.

        Notes
        -----
        It is assumed, that between support points, there is a drift effect shifting the
        meassurements in a way, that can be described, by a model function M(t, p, origin, target).
        (With 0<=t<=1, p being a parameter set, and origin, target being floats).

        Note, that its possible for the model to have no free parameters p at all. (linear drift mainly)

        The drift model, directly after the last support point (t=0),
        should evaluate to the origin - calibration level (origin), and directly before the next
        support point (t=1), it should evaluate to the target calibration level (target).


            M(0, p, origin, target) = origin
            M(1, p, origin, target) = target


        The model is than fitted to any data chunk in between support points, by optimizing
        the parameters p, and thus, obtaining optimal parameterset P.

        The new values at t are computed via:::

            new_vals(t) = old_vals(t) + M(t, P, origin, target) - M_drift(t, P, origin, new_target)

        Wheras ``new_target`` represents the value level immediately after the next support point.

        Examples
        --------
        Some examples of meaningful driftmodels.

        Linear drift modell (no free parameters).


        >>> Model = lambda t, origin, target: origin + t*target

        exponential drift model (exponential raise!)

        >>> expFunc = lambda t, a, b, c: a + b * (np.exp(c * x) - 1)
        >>> Model = lambda t, p, origin, target: expFunc(t, (target - origin) / (np.exp(abs(c)) - 1), abs(c))

        Exponential and linear driftmodels are part of the ``ts_operators`` library, under the names
        ``expDriftModel`` and ``linearDriftModel``.
        """
        pass

    def correctRegimeAnomaly(self, field, cluster_field, model, tolerance, epoch):
        """
        Function fits the passed model to the different regimes in data[field] and tries to correct
        those values, that have assigned a negative label by data[cluster_field].

        Currently, the only correction mode supported is the "parameter propagation."

        This means, any regime :math:`z`, labeled negatively and being modeled by the parameters p, gets corrected via:

        :math:`z_{correct} = z + (m(p^*) - m(p))`,

        where :math:`p^*` denotes the parameter set belonging to the fit of the nearest not-negatively labeled cluster.

        Parameters
        ----------
        field : str
            The fieldname of the data column, you want to correct.
        cluster_field : str
            A string denoting the field in data, holding the cluster label for the data you want to correct.
        model : Callable
            The model function to be fitted to the regimes.
            It must be a function of the form :math:`f(x, *p)`, where :math:`x` is the ``numpy.array`` holding the
            independent variables and :math:`p` are the model parameters that are to be obtained by fitting.
            Depending on the `x_date` parameter, independent variable x will either be the timestamps
            of every regime transformed to seconds from epoch, or it will be just seconds, counting the regimes length.
        tolerance : {None, str}, default None:
            If an offset string is passed, a data chunk of length `offset` right at the
            start and right at the end is ignored when fitting the model. This is to account for the
            unreliability of data near the changepoints of regimes.
        epoch : bool, default False
            If True, use "seconds from epoch" as x input to the model func, instead of "seconds from regime start".
        """
        pass

    def correctOffset(
        self,
    ):
        """
        Parameters
        ----------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        field : str
            The fieldname of the data column, you want to correct.
        flags : saqc.Flags
            Container to store flags of the data.
        max_jump : float
            when searching for changepoints in mean - this is the threshold a mean difference in the
            sliding window search must exceed to trigger changepoint detection.
        spread : float
            threshold denoting the maximum, regimes are allowed to abolutely differ in their means
            to form the "normal group" of values.
        window : str
            Size of the adjacent windows that are used to search for the mean changepoints.
        min_periods : int
            Minimum number of periods a search window has to contain, for the result of the changepoint
            detection to be considered valid.
        tolerance : {None, str}, default None:
            If an offset string is passed, a data chunk of length `offset` right from the
            start and right before the end of any regime is ignored when calculating a regimes mean for data correcture.
            This is to account for the unrelyability of data near the changepoints of regimes.
        """
        pass

    def flagRegimeAnomaly(
        self, field, cluster_field, spread, method, metric, frac, flag
    ):
        """
        Flags anomalous regimes regarding to modelling regimes of field.

        "Normality" is determined in terms of a maximum spreading distance,
        regimes must not exceed in respect to a certain metric and linkage method.

        In addition, only a range of regimes is considered "normal", if it models
        more then `frac` percentage of the valid samples in "field".

        Note, that you must detect the regime changepoints prior to calling this function.

        Note, that it is possible to perform hypothesis tests for regime equality
        by passing the metric a function for p-value calculation and selecting linkage
        method "complete".

        Parameters
        ----------
        field : str
            Name of the column to process
        cluster_field : str
            Column in data, holding the cluster labels for the samples in field.
            (has to be indexed equal to field)
        spread : float
            A threshold denoting the value level, up to wich clusters a agglomerated.
        method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
            The linkage method for hierarchical (agglomerative) clustering of the variables.
        metric : Callable, default lambda x,y: np.abs(np.nanmean(x) - np.nanmean(y))
            A metric function for calculating the dissimilarity between 2 regimes.
            Defaults to the difference in mean.
        frac : float
            Has to be in [0,1]. Determines the minimum percentage of samples,
            the "normal" group has to comprise to be the normal group actually.
        flag : float, default BAD
            flag to set.
        """
        pass

    def assignRegimeAnomaly(self, field, cluster_field, spread, method, metric, frac):
        """
        A function to detect values belonging to an anomalous regime regarding modelling
        regimes of field.

        The function changes the value of the regime cluster labels to be negative.
        "Normality" is determined in terms of a maximum spreading distance, regimes must
        not exceed in respect to a certain metric and linkage method. In addition,
        only a range of regimes is considered "normal", if it models more then `frac`
        percentage of the valid samples in "field". Note, that you must detect the regime
        changepoints prior to calling this function. (They are expected to be stored
        parameter `cluster_field`.)

        Note, that it is possible to perform hypothesis tests for regime equality by
        passing the metric a function for p-value calculation and selecting linkage
        method "complete".

        Parameters
        ----------
        field : str
            Name of the column to process
        cluster_field : str
            Column in data, holding the cluster labels for the samples in field.
            (has to be indexed equal to field)
        spread : float
            A threshold denoting the value level, up to wich clusters a agglomerated.
        method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
            The linkage method for hierarchical (agglomerative) clustering of the variables.
        metric : Callable, default lambda x,y: np.abs(np.nanmean(x) - np.nanmean(y))
            A metric function for calculating the dissimilarity between 2 regimes.
            Defaults to the difference in mean.
        frac : float
            Has to be in [0,1]. Determines the minimum percentage of samples,
            the "normal" group has to comprise to be the normal group actually.
        """
        pass

    def forceFlags(self, field, flag, kwargs):
        """
        Set whole column to a flag value.

        Parameters
        ----------
        field : str
            columns name that holds the data
        flag : float, default BAD
            flag to set
        kwargs : dict
            unused

        See Also
        --------
        clearFlags : set whole column to UNFLAGGED
        flagUnflagged : set flag value at all unflagged positions
        """
        pass

    def clearFlags(self, field, kwargs):
        """
        Set whole column to UNFLAGGED.

        Parameters
        ----------
        field : str
            columns name that holds the data
        kwargs : dict
            unused

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the data is not relevant
        for processing.
        A warning is triggered if the ``flag`` keyword is given, because the flags are
        always set to `UNFLAGGED`.


        See Also
        --------
        forceFlags : set whole column to a flag value
        flagUnflagged : set flag value at all unflagged positions
        """
        pass

    def flagUnflagged(self, field, flag, kwargs):
        """
        Function sets a flag at all unflagged positions.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        flag : float, default BAD
            flag value to set
        kwargs : Dict
            unused

        Notes
        -----
        This function ignores the ``dfilter`` keyword, because the data is not relevant
        for processing.

        See Also
        --------
        clearFlags : set whole column to UNFLAGGED
        forceFlags : set whole column to a flag value
        """
        pass

    def flagManual(self, field, mdata, method, mformat, mflag, flag):
        """
        Flag data by given, "manually generated" data.

        The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
        The format of mdata can be an indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
        but also can be a plain list- or array-like.
        How indexed mdata is aligned to data is specified via the `method` parameter.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        mdata : pd.Series, pd.DataFrame, DictOfSeries, str, list or np.ndarray
            The Data determining, wich intervals are to be flagged, or a string, denoting under which field the data is
            accessable.
        method : {'plain', 'ontime', 'left-open', 'right-open', 'closed'}, default 'plain'
            Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
            index.

            * 'plain': mdata must have the same length as data and is projected one-to-one on data.
            * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
            * 'right-open': mdata defines intervals, values are to be projected on.
              The intervals are defined,

              (1) Either, by any two consecutive timestamps t_1 and 1_2 where t_1 is valued with mflag, or by a series,
              (2) Or, a Series, where the index contains in the t1 timestamps nd the values the respective t2 stamps.

              The value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.

            * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.
            * 'closed': like 'right-open', but the projected interval now covers all t with t_1 <= t <= t_2.

        mformat : {"start-end", "mflag"}, default "start-end"

            * "start-end": mdata is a Series, where every entry indicates an interval to-flag. The index defines the left
              bound, the value defines the right bound.
            * "mflag": mdata is an array like, with entries containing 'mflag',where flags shall be set. See documentation
              for examples.

        mflag : scalar
            The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
        flag : float, default BAD
            flag to set.

        Examples
        --------
        An example for mdata

        .. doctest:: ExampleFlagManual

           >>> mdata = pd.Series([1,0,1], index=pd.to_datetime(['2000-02', '2000-03', '2001-05']))
           >>> mdata
           2000-02-01    1
           2000-03-01    0
           2001-05-01    1
           dtype: int64

        On *dayly* data, with the 'ontime' method, only the provided timestamps are used.
        Bear in mind that only exact timestamps apply, any offset will result in ignoring
        the timestamp.

        .. doctest:: ExampleFlagManual

           >>> data = a=pd.Series(0, index=pd.date_range('2000-01-31', '2000-03-02', freq='1D'), name='dailyData')
           >>> qc = saqc.SaQC(data)
           >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='ontime')
           >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
           2000-01-31    False
           2000-02-01    True
           2000-02-02    False
           2000-02-03    False
           ..            ..
           2000-02-29    False
           2000-03-01    True
           2000-03-02    False
           Freq: D, dtype: bool

        With the 'right-open' method, the mdata is forward fill:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='right-open')
           >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
           2000-01-31    False
           2000-02-01    True
           2000-02-02    True
           ..            ..
           2000-02-29    True
           2000-03-01    False
           2000-03-02    False
           Freq: D, dtype: bool

        With the 'left-open' method, backward filling is used:

        .. doctest:: ExampleFlagManual

           >>> qc = qc.flagManual('dailyData', mdata, mflag=1, mformat='mdata', method='left-open')
           >>> qc.flags['dailyData'] > UNFLAGGED #doctest:+SKIP
           2000-01-31    False
           2000-02-01    False
           2000-02-02    True
           ..            ..
           2000-02-29    True
           2000-03-01    True
           2000-03-02    False
           Freq: D, dtype: bool
        """
        pass

    def flagDummy(self, field):
        """
        Function does nothing but returning data and flags.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        """
        pass

    def processGeneric(self, field, func):
        """
        Generate/process data with user defined functions.

        Formally, what the function does, is the following:

        1.  Let F be a Callable, depending on fields f_1, f_2,...f_K, (F = F(f_1, f_2,...f_K))
            Than, for every timestamp t_i that occurs in at least one of the timeseries data[f_j] (outer join),
            The value v_i is computed via:
            v_i = data([f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]), if all data[f_j][t_i] do exist
            v_i = ``np.nan``, if at least one of the data[f_j][t_i] is missing.
        2.  The result is stored to ``data[target]``, if ``target`` is given or to ``data[field]`` otherwise

        Parameters
        ----------
        field : str or list of str
            The variable(s) passed to func.
        func : callable
            Function to call on the variables given in ``field``. The return value will be written
            to ``target`` or ``field`` if the former is not given. This implies, that the function
            needs to accept the same number of arguments (of type pandas.Series) as variables given
            in ``field`` and should return an iterable of array-like objects with the same number
            of elements as given in ``target`` (or ``field`` if ``target`` is not specified).
        target: str or list of str
            The variable(s) to write the result of ``func`` to. If not given, the variable(s)
            specified in ``field`` will be overwritten. If a ``target`` is not given, it will be
            created.
        flag: float, default ``UNFLAGGED``
            The quality flag to set. The default ``UNFLAGGED`` states the general idea, that
            ``processGeneric`` generates 'new' data without direct relation to the potentially
            already present flags.
        dfilter: float, default ``FILTER_ALL``
            Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
            data value is inappropiate for further usage.

        Examples
        --------
        Compute the sum of the variables 'rainfall' and 'snowfall' and save the result to
        a (new) variable 'precipitation'

        .. testsetup::

           qc = saqc.SaQC(pd.DataFrame({'rainfall':[0], 'snowfall':[0], 'precipitation':[0]}, index=pd.DatetimeIndex([0])))


        >>> qc = qc.processGeneric(field=["rainfall", "snowfall"], target="precipitation'", func=lambda x, y: x + y)
        """
        pass

    def flagGeneric(self, field, func):
        """
        Flag data with user defined functions.

        Formally, what the function does, is the following:
        Let X be a Callable, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
        Than for every timestamp t_i in data[field]:
        data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.

        Parameters
        ----------
        field : str or list of str
            The variable(s) passed to func.
        func : callable
            Function to call on the variables given in ``field``. The function needs to accept the same
            number of arguments (of type pandas.Series) as variables given in ``field`` and return an
            iterable of array-like objects of with dtype bool and with the same number of elements as
            given in ``target`` (or ``field`` if ``target`` is not specified). The function output
            determines the values to flag.
        target: str or list of str
            The variable(s) to write the result of ``func`` to. If not given, the variable(s)
            specified in ``field`` will be overwritten. If a ``target`` is not given, it will be
            created.
        flag: float, default ``UNFLAGGED``
            The quality flag to set. The default ``UNFLAGGED`` states the general idea, that
            ``processGeneric`` generates 'new' data without direct relation to the potentially
            already present flags.
        dfilter: float, default ``FILTER_ALL``
            Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
            data value is inappropiate for further usage.

        Examples
        --------

        .. testsetup:: exampleFlagGeneric

           qc = saqc.SaQC(pd.DataFrame({'temperature':[0], 'uncertainty':[0], 'rainfall':[0], 'fan':[0]}, index=pd.DatetimeIndex([0])))

        1. Flag the variable 'rainfall', if the sum of the variables 'temperature' and 'uncertainty' is below zero:

        .. testcode:: exampleFlagGeneric

           qc.flagGeneric(field=["temperature", "uncertainty"], target="rainfall", func= lambda x, y: x + y < 0)

        2. Flag the variable 'temperature', where the variable 'fan' is flagged:

        .. testcode:: exampleFlagGeneric

           qc.flagGeneric(field="fan", target="temperature", func=lambda x: isflagged(x))

        3. The generic functions also support all pandas and numpy functions:

        .. testcode:: exampleFlagGeneric

           qc = qc.flagGeneric(field="fan", target="temperature", func=lambda x: np.sqrt(x) < 7)
        """
        pass

    def interpolateByRolling(self, field, window, func, center, min_periods, flag):
        """
        Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        window : int, str
            The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
            an string is interpreted as an offset. ( see `pandas.rolling` for more information).
            Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.

        func : Callable
            The function used for aggregation.

        center : bool, default True
            Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.

        min_periods : int
            Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
            computed.

        flag : float or None, default UNFLAGGED
            Flag that is to be inserted for the interpolated values.
            If `None` the old flags are kept, even if the data is valid now.
        """
        pass

    def interpolateInvalid(self, field, method, order, limit, flag, downgrade):
        """
        Function to interpolate nan values in the data.

        There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
        the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
            "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
            The interpolation method to use.

        order : int, default 2
            If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
            order.

        limit : int, default 2
            Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
            interpolation to chunks, containing not more than `limit` successive nan entries.

        flag : float or None, default UNFLAGGED
            Flag that is set for interpolated values. If ``None``, no flags are set at all.

        downgrade : bool, default False
            If `True` and the interpolation can not be performed at current order, retry with a lower order.
            This can happen, because the chosen ``method`` does not support the passed ``order``, or
            simply because not enough values are present in a interval.
        """
        pass

    def interpolateIndex(self, field, freq, method, order, limit, downgrade):
        """
        Function to interpolate the data at regular (equidistant) timestamps (or Grid points).

        Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
        valid data value within "freq" range.

        Parameters
        ----------
        field : str
            Name of the column, holding the data-to-be-interpolated.

        freq : str
            An Offset String, interpreted as the frequency of
            the grid you want to interpolate your data at.

        method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
            "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
            The interpolation method you want to apply.

        order : int, default 2
            If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
            order.

        limit : int, default 2
            Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
            interpolation to chunks, containing not more than `limit` successive nan entries.

        downgrade : bool, default False
            If `True` and the interpolation can not be performed at current order, retry with a lower order.
            This can happen, because the chosen ``method`` does not support the passed ``order``, or
            simply because not enough values are present in a interval.

        """
        pass

    def flagByStatLowPass(self, field, flag):
        """
        Flag *chunks* of length, `window`:

        1. If they excexceed `thresh` with regard to `stat`:
        2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_window`,
           `excexceed `sub_thresh` with regard to `stat`:

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        flag : float, default BAD
            flag to set

        Returns
        -------
        """
        pass

    def flagByStray(self, field, freq, min_periods, iter_start, alpha, flag):
        """
        Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.

        Find more information on the algorithm in References [1].

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        freq : str, int, or None, default None
            Determines the segmentation of the data into partitions, the kNN algorithm is
            applied onto individually.

            * ``np.inf``: Apply Scoring on whole data set at once
            * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
            * Offset String : Apply scoring on successive partitions of temporal extension
              matching the passed offset string

        min_periods : int, default 11
            Minimum number of periods per partition that have to be present for a valid
            outlier dettection to be made in this partition. (Only of effect, if `freq`
            is an integer.) Partition min value must always be greater then the
            nn_neighbors value.

        iter_start : float, default 0.5
            Float in [0,1] that determines which percentage of data is considered
            "normal". 0.5 results in the stray algorithm to search only the upper 50 % of
            the scores for the cut off point. (See reference section for more information)

        alpha : float, default 0.05
            Level of significance by which it is tested, if a score might be drawn from
            another distribution, than the majority of the data.

        flag : float, default BAD
            flag to set.

        References
        ----------
        [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in
            high dimensional data. arXiv preprint arXiv:1908.04000.
        """
        pass

    def flagMVScores(
        self,
        field,
        trafo,
        alpha,
        n,
        func,
        iter_start,
        partition,
        partition_min,
        partition_trafo,
        stray_range,
        drop_flagged,
        thresh,
        min_periods,
        target,
        flag,
    ):
        """
                The algorithm implements a 3-step outlier detection procedure for simultaneously
                flagging of higher dimensional data (dimensions > 3).

                In references [1], the procedure is introduced and exemplified with an
                application on hydrological data. See the notes section for an overview over the
                algorithms basic steps.

                Parameters
                ----------
                field : list of str
                    List of fieldnames, corresponding to the variables that are to be included
                    into the flagging process.

                trafo : callable, default lambda x:x
                    Transformation to be applied onto every column before scoring. Will likely
                    get deprecated soon. Its better to transform the data in a processing step,
                    preceeeding the call to ``flagMVScores``.

                alpha : float, default 0.05
                    Level of significance by which it is tested, if an observations score might
                    be drawn from another distribution than the majority of the observation.

                n : int, default 10
                    Number of neighbors included in the scoring process for every datapoint.

                func : Callable[numpy.array, float], default np.sum
                    The function that maps the set of every points k-nearest neighbor distances
                    onto a certain scoring.

                iter_start : float, default 0.5
                    Float in [0,1] that determines which percentage of data is considered
                    "normal". 0.5 results in the threshing algorithm to search only the upper 50
                    % of the scores for the cut off point. (See reference section for more
                    information)

                partition : {None, str, int}, default None
                    Only effective when `threshing` = 'stray'. Determines the size of the data
                    partitions, the data is decomposed into. Each partition is checked seperately
                    for outliers. If a String is passed, it has to be an offset string and it
                    results in partitioning the data into parts of according temporal length. If
                    an integer is passed, the data is simply split up into continous chunks of
                    `freq` periods. if ``None`` is passed (default), all the data will be tested
                    in one run.

                partition_min : int, default 11
                    Only effective when `threshing` = 'stray'. Minimum number of periods per
                    partition that have to be present for a valid outlier detection to be made in
                    this partition. (Only of effect, if `stray_partition` is an integer.)

                partition_trafo : bool, default True
                    Whether or not to apply the passed transformation on every partition the
                    algorithm is applied on, separately.

                stray_range : {None, str}, default None
                    If not None, it is tried to reduce the stray result onto single outlier
                    components of the input fields. An offset string, denoting the range of the
                    temporal surrounding to include into the MAD testing while trying to reduce
                    flags.

                drop_flagged : bool, default False
                    Only effective when `range` is not ``None``. Whether or not to drop flagged
                    values other than the value under test from the temporal surrounding before
                    checking the value with MAD.

                thresh : float, default 3.5
                    Only effective when `range` is not ``None``. The `critical` value,
                    controlling wheather the MAD score is considered referring to an outlier or
                    not. Higher values result in less rigid flagging. The default value is widely
                    considered apropriate in the literature.

                min_periods : int, 1
                    Only effective when `range` is not ``None``. Minimum number of meassurements
                    necessarily present in a reduction interval for reduction actually to be
                    performed.
        <<<<<<< HEAD

                target : None
                    ignored.

        =======

                target : list of str
                    List of field names to write the output to, these fields should not already
                    exist.

        >>>>>>> 7a15bb35746a5cb984a850ea45df42d37f411e82
                flag : float, default BAD
                    flag to set.

                Notes
                -----
                The basic steps are:

                1. transforming

                The different data columns are transformed via timeseries transformations to
                (a) make them comparable and
                (b) make outliers more stand out.

                This step is usually subject to a phase of research/try and error. See [1] for more
                details.

                Note, that the data transformation as an built-in step of the algorithm,
                will likely get deprecated soon. Its better to transform the data in a processing
                step, preceeding the multivariate flagging process. Also, by doing so, one gets
                mutch more control and variety in the transformation applied, since the `trafo`
                parameter only allows for application of the same transformation to all of the
                variables involved.

                2. scoring

                Every observation gets assigned a score depending on its k nearest neighbors. See
                the `scoring_method` parameter description for details on the different scoring
                methods. Furthermore [1], [2] may give some insight in the pro and cons of the
                different methods.

                3. threshing

                The gaps between the (greatest) scores are tested for beeing drawn from the same
                distribution as the majority of the scores. If a gap is encountered, that,
                with sufficient significance, can be said to not be drawn from the same
                distribution as the one all the smaller gaps are drawn from, than the observation
                belonging to this gap, and all the observations belonging to gaps larger then
                this gap, get flagged outliers. See description of the `threshing` parameter for
                more details. Although [2] gives a fully detailed overview over the `stray`
                algorithm.
        """
        pass

    def flagRaise(
        self,
        field,
        thresh,
        raise_window,
        freq,
        average_window,
        raise_factor,
        slope,
        weight,
        flag,
    ):
        """
        The function flags raises and drops in value courses, that exceed a certain threshold
        within a certain timespan.

        The parameter variety of the function is owned to the intriguing
        case of values, that "return" from outlierish or anomalious value levels and
        thus exceed the threshold, while actually being usual values.

        NOTE, the dataset is NOT supposed to be harmonized to a time series with an
        equidistant frequency grid.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        thresh : float
            The threshold, for the total rise (thresh > 0), or total drop (thresh < 0),
            value courses must not exceed within a timespan of length `raise_window`.
        raise_window : str
            An offset string, determining the timespan, the rise/drop thresholding refers
            to. Window is inclusively defined.
        freq : str
            An offset string, determining The frequency, the timeseries to-be-flagged is
            supposed to be sampled at. The window is inclusively defined.
        average_window : {None, str}, default None
            See condition (2) of the description linked in the references. Window is
            inclusively defined. The window defaults to 1.5 times the size of `raise_window`
        raise_factor : float, default 2
            See second condition listed in the notes below.
        slope : {None, float}, default None
            See third condition listed in the notes below.
        weight : float, default 0.8
            See third condition listed in the notes below.
        flag : float, default BAD
            flag to set.

        Notes
        -----
        The value :math:`x_{k}` of a time series :math:`x` with associated
        timestamps :math:`t_i`, is flagged a raise, if:

        * There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within `raise_window`
          range, so that:

          * :math:`M = |x_k - x_s | >`  `thresh` :math:`> 0`

        * The weighted average :math:`\mu^{*}` of the values, preceding :math:`x_{k}`
          within `average_window`
          range indicates, that :math:`x_{k}` does not return from an "outlierish" value
          course, meaning that:

          * :math:`x_k > \mu^* + ( M` / `mean_raise_factor` :math:`)`

        * Additionally, if ``min_slope`` is not `None`, :math:`x_{k}` is checked for being
          sufficiently divergent from its very predecessor :math:`x_{k-1}`, meaning that, it
          is additionally checked if:

          * :math:`x_k - x_{k-1} >` `min_slope`
          * :math:`t_k - t_{k-1} >` `weight` :math:`\times` `freq`
        """
        pass

    def flagMAD(self, field, window, flag):
        """
        The function represents an implementation of the modyfied Z-score outlier detection method.

        See references [1] for more details on the algorithm.

        Note, that the test needs the input data to be sampled regularly (fixed sampling rate).

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
        window : str
           Offset string. Denoting the windows size that the "Z-scored" values have to lie in.
        z: float, default 3.5
            The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
        flag : float, default BAD
            flag to set.

        References
        ----------
        [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        """
        pass

    def flagOffset(self, field, thresh, tolerance, window, thresh_relative, flag):
        """
        A basic outlier test that work on regular and irregular sampled data

        The test classifies values/value courses as outliers by detecting not only a rise
        in value, but also, checking for a return to the initial value level.

        Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with
        associated timestamps :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if

        1. :math:`|x_{n-1} - x_{n + s}| >` `thresh`, for all :math:`s \in [0,1,2,...,k]`

        2. :math:`|x_{n-1} - x_{n+k+1}| <` `tolerance`

        3. :math:`|t_{n-1} - t_{n+k+1}| <` `window`

        Note, that this definition of a "spike" not only includes one-value outliers, but
        also plateau-ish value courses.

        Parameters
        ----------
        field : str
            The field in data.
        thresh : float
            Minimum difference between to values, to consider the latter one as a spike. See condition (1)
        tolerance : float
            Maximum difference between pre-spike and post-spike values. See condition (2)
        window : {str, int}, default '15min'
            Maximum length of "spiky" value courses. See condition (3). Integer defined window length are only allowed for
            regularly sampled timeseries.
        thresh_relative : {float, None}, default None
            Relative threshold.
        flag : float, default BAD
            flag to set.

        References
        ----------
        The implementation is a time-window based version of an outlier test from the UFZ Python library,
        that can be found here:

        https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py
        """
        pass

    def flagByGrubbs(self, field, window, alpha, min_periods, flag):
        """
        The function flags values that are regarded outliers due to the grubbs test.

        See reference [1] for more information on the grubbs tests definition.

        The (two-sided) test gets applied onto data chunks of size "window". The tests
        application  will be iterated on each data-chunk under test, till no more
        outliers are detected in that chunk.

        Note, that the test performs poorely for small data chunks (resulting in heavy
        overflagging). Therefor you should select "window" so that every window contains
        at least > 8 values and also adjust the min_periods values accordingly.

        Note, that the data to be tested by the grubbs test are expected to be distributed
        "normalish".

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        window : {int, str}
            The size of the window you want to use for outlier testing. If an integer is
            passed, the size refers to the number of periods of every testing window. If a
            string is passed, it has to be an offset string, and will denote the total
            temporal extension of every window.
        alpha : float, default 0.05
            The level of significance, the grubbs test is to be performed at. (between 0 and 1)
        min_periods : int, default 8
            The minimum number of values that have to be present in an interval under test,
            for a grubbs test result to be accepted. Only makes sence in case `window` is
            an offset string.
        pedantic: boolean, default False
            If True, every value gets checked twice for being an outlier. Ones in the
            initial rolling window and one more time in a rolling window that is lagged
            by half the windows delimeter (window/2). Recommended for avoiding false
            positives at the window edges. Only available when rolling with integer
            defined window size.
        flag : float, default BAD
            flag to set.

        References
        ----------
        introduction to the grubbs test:

        [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
        """
        pass

    def flagRange(self, field, min, max, flag):
        """
        Function flags values not covered by the closed interval [`min`, `max`].

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-flagged.
        min : float
            Lower bound for valid data.
        max : float
            Upper bound for valid data.
        flag : float, default BAD
            flag to set.
        """
        pass

    def flagCrossStatistic(self, field, thresh, method, flag, target):
        """
        Function checks for outliers relatively to the "horizontal" input data axis.

        For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
        for outlier detection:

        1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
           :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
        2. for every :math:`0 <= i <= K`, the value
           :math:`m_j = median(\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\})` is calculated
        3. for every :math:`0 <= i <= K`, the set
           :math:`\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\}` is tested for outliers with the
           specified method (`cross_stat` parameter).

        Parameters
        ----------
        field : list of str
            List of fieldnames in data, determining wich variables are to be included into the flagging process.
        thresh : float
            Threshold which the outlier score of an value must exceed, for being flagged an outlier.
        method : {'modZscore', 'Zscore'}, default 'modZscore'
            Method used for calculating the outlier scores.

            * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
            * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
              See References [1]

        flag : float, default BAD
            flag to set.

        target : None
            ignored


        References
        ----------
        [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        """
        pass

    def flagPatternByWavelet(self, field):
        """
        Pattern recognition via wavelets.

        The steps are:
         1. work on chunks returned by a moving window
         2. each chunk is compared to the given pattern, using the wavelet algorithm as
            presented in [1]
         3. if the compared chunk is equal to the given pattern it gets flagged

        Parameters
        ----------

        field : str
            The fieldname of the data column, you want to correct.
        """
        pass

    def calculateDistanceByDTW(self, reference, normalize):
        """
        Calculate the DTW-distance of data to pattern in a rolling calculation.

        The data is compared to pattern in a rolling window.
        The size of the rolling window is determined by the timespan defined
        by the first and last timestamp of the reference data's datetime index.

        For details see the linked functions in the `See Also` section.

        Parameters
        ----------
        reference : : pd.Series
            Reference series. Must have datetime-like index, must not contain NaNs
            and must not be empty.

        forward: bool, default True
            If `True`, the distance value is set on the left edge of the data chunk. This
            means, with a perfect match, `0.0` marks the beginning of the pattern in
            the data. If `False`, `0.0` would mark the end of the pattern.

        normalize : bool, default True
            If `False`, return unmodified distances.
            If `True`, normalize distances by the number of observations in the reference.
            This helps to make it easier to find a good cutoff threshold for further
            processing. The distances then refer to the mean distance per datapoint,
            expressed in the datas units.

        Notes
        -----
        The data must be regularly sampled, otherwise a ValueError is raised.
        NaNs in the data will be dropped before dtw distance calculation.

        See Also
        --------
        flagPatternByDTW : flag data by DTW
        """
        pass

    def flagPatternByDTW(self, field, reference, max_distance, normalize):
        """
        Pattern Recognition via Dynamic Time Warping.

        The steps are:
         1. work on a moving window
         2. for each data chunk extracted from each window, a distance to the given pattern
            is calculated, by the dynamic time warping algorithm [1]
         3. if the distance is below the threshold, all the data in the window gets flagged

        Parameters
        ----------
        field : str
            The name of the data column

        reference : str
            The name in `data` which holds the pattern. The pattern must not have NaNs,
            have a datetime index and must not be empty.

        max_distance : float, default 0.0
            Maximum dtw-distance between chunk and pattern, if the distance is lower than
            ``max_distance`` the data gets flagged. With default, ``0.0``, only exact
            matches are flagged.

        normalize : bool, default True
            If `False`, return unmodified distances.
            If `True`, normalize distances by the number of observations of the reference.
            This helps to make it easier to find a good cutoff threshold for further
            processing. The distances then refer to the mean distance per datapoint,
            expressed in the datas units.

        plot: bool, default False
            Show a calibration plot, which can be quite helpful to find the right threshold
            for `max_distance`. It works best with `normalize=True`. Do not use in automatic
            setups / pipelines. The plot show three lines:

            - data: the data the function was called on
            - distances: the calculated distances by the algorithm
            - indicator: have to distinct levels: `0` and the value of `max_distance`.
              If `max_distance` is `0.0` it defaults to `1`. Everywhere where the
              indicator is not `0` the data will be flagged.

        Notes
        -----
        The window size of the moving window is set to equal the temporal extension of the
        reference datas datetime index.

        References
        ----------
        Find a nice description of underlying the Dynamic Time Warping Algorithm here:

        [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
        """
        pass

    def linear(self, field, freq):
        """
        A method to "regularize" data by interpolating linearly the data at regular timestamp.

        A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

        Interpolated values will get assigned the worst flag within freq-range.

        Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
        not-na) datapoint preceeding them and one succeeding them within freq range.
        Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
        ``UNFLAGGED``.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-regularized.

        freq : str
            An offset string. The frequency of the grid you want to interpolate your data at.
        """
        pass

    def interpolate(self, field, freq, method, order):
        """
        A method to "regularize" data by interpolating the data at regular timestamp.

        A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

        Interpolated values will get assigned the worst flag within freq-range.

        There are available all the interpolations from the pandas.Series.interpolate method and they are called by
        the very same keywords.

        Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
        and NOT ``'linear'``.

        Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
        not-na) datapoint preceeding them and one succeeding them within freq range.
        Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
        ``UNFLAGGED``.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-regularized.

        freq : str
            An offset string. The frequency of the grid you want to interpolate your data at.

        method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
            "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
            The interpolation method you want to apply.

        order : int, default 1
            If your selected interpolation method can be performed at different *orders* - here you pass the desired
            order.
        """
        pass

    def shift(self, field, freq, method, freq_check):
        """
        Function to shift data and flags to a regular (equidistant) timestamp grid, according to ``method``.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-shifted.

        freq : str
            An frequency Offset String that will be interpreted as the sampling rate you want the data to be shifted to.

        method : {'fshift', 'bshift', 'nshift'}, default 'nshift'
            Specifies how misaligned data-points get propagated to a grid timestamp.
            Following choices are available:

            * 'nshift' : every grid point gets assigned the nearest value in its range. (range = +/- 0.5 * `freq`)
            * 'bshift' : every grid point gets assigned its first succeeding value, if one is available in
              the succeeding sampling interval.
            * 'fshift' : every grid point gets assigned its ultimately preceding value, if one is available in
              the preceeding sampling interval.

        freq_check : {None, 'check', 'auto'}, default None

            * ``None`` : do not validate frequency-string passed to `freq`
            * 'check' : estimate frequency and log a warning if estimate miss matches frequency string passed to `freq`,
              or if no uniform sampling rate could be estimated
            * 'auto' : estimate frequency and use estimate. (Ignores `freq` parameter.)
        """
        pass

    def resample(
        self,
        field,
        freq,
        func,
        maxna,
        maxna_group,
        maxna_flags,
        maxna_group_flags,
        flag_func,
        freq_check,
    ):
        """
        Function to resample the data.

        The data will be sampled at regular (equidistant) timestamps aka. Grid points.
        Sampling intervals therefore get aggregated with a function, specified by
        'agg_func' parameter and the result gets projected onto the new timestamps with a
        method, specified by "method". The following method (keywords) are available:

        * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get
            aggregated with agg_func and assigned to it.
        * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and
            the result gets assigned to the last grid point.
        * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and
            the result gets assigned to the next grid point.


        Note, that. if possible, functions passed to agg_func will get projected
        internally onto pandas.resample methods, wich results in some reasonable
        performance boost - however, for this to work, you should pass functions that
        have the __name__ attribute initialised and the according methods name assigned
        to it. Furthermore, you shouldnt pass numpys nan-functions (``nansum``,
        ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and
        they will thus not trigger ``resample.func()``, but the slower ``resample.apply(
        nanfunc)``. Also, internally, no nans get passed to the functions anyway,
        so that there is no point in passing the nan functions.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-resampled.

        freq : str
            An Offset String, that will be interpreted as the frequency you want to
            resample your data with.

        func : Callable
            The function you want to use for aggregation.

        method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
            Specifies which intervals to be aggregated for a certain timestamp. (preceding,
            succeeding or "surrounding" interval). See description above for more details.

        maxna : {None, int}, default None
            Maximum number NaNs in a resampling interval. If maxna is exceeded, the interval
            is set entirely to NaN.

        maxna_group : {None, int}, default None
            Same as `maxna` but for consecutive NaNs.

        maxna_flags : {None, int}, default None
            Same as `max_invalid`, only applying for the flags. The flag regarded
            as "invalid" value, is the one passed to empty_intervals_flag (
            default=``BAD``). Also this is the flag assigned to invalid/empty intervals.

        maxna_group_flags : {None, int}, default None
            Same as `maxna_flags`, only applying onto flags. The flag regarded as
            "invalid" value, is the one passed to empty_intervals_flag. Also this is the
            flag assigned to invalid/empty intervals.

        flag_func : Callable, default: max
            The function you want to aggregate the flags with. It should be capable of
            operating on the flags dtype (usually ordered categorical).

        freq_check : {None, 'check', 'auto'}, default None

            * ``None``: do not validate frequency-string passed to `freq`
            * ``'check'``: estimate frequency and log a warning if estimate miss matchs
                frequency string passed to 'freq', or if no uniform sampling rate could be
                estimated
            * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)
        """
        pass

    def concatFlags(self, field, target, method, freq, drop):
        """
        The Function appends flags history of ``fields`` to flags history of ``target``.
        Before Appending, columns in ``field`` history are projected onto the target index via ``method``

        method: (field_flag in associated with "field", source_flags associated with "source")

        'inverse_nagg' - all target_flags within the range +/- freq/2 of a field_flag, get assigned this field flags value.
            (if field_flag > target_flag)
        'inverse_bagg' - all target_flags succeeding a field_flag within the range of "freq", get assigned this field flags
            value. (if field_flag > target_flag)
        'inverse_fagg' - all target_flags preceeding a field_flag within the range of "freq", get assigned this field flags
            value. (if field_flag > target_flag)

        'inverse_interpolation' - all target_flags within the range +/- freq of a field_flag, get assigned this source flags value.
            (if field_flag > target_flag)

        'inverse_nshift' - That target_flag within the range +/- freq/2, that is nearest to a field_flag, gets the source
            flags value. (if field_flag > target_flag)
        'inverse_bshift' - That target_flag succeeding a field flag within the range freq, that is nearest to a
            field_flag, gets assigned this field flags value. (if field_flag > target_flag)
        'inverse_nshift' - That target_flag preceeding a field flag within the range freq, that is nearest to a
            field_flag, gets assigned this field flags value. (if field_flag > target_flag)

        'match' - any target_flag with a timestamp matching a field_flags timestamp gets this field_flags value
        (if field_flag > target_flag)

        Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
        you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.

        Parameters
        ----------
        field : str
            Fieldname of flags history to append.

        target : str
            Field name of flags history to append to.

        method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
                 'match'}
            The method used for projection of ``field`` flags onto ``target`` flags. See description above for more details.

        freq : {None, str},default None
            The ``freq`` determines the projection range for the projection method. See above description for more details.
            Defaultly (None), the sampling frequency of ``field`` is used.

        drop : default False
            If set to `True`, the `field` column will be removed after processing
        """
        pass

    def calculatePolynomialResidues(self, field, window, order, min_periods):
        """
        Fits a polynomial model to the data and calculate the residues.

        The residue  is calculated by fitting a polynomial of degree `order` to a data
        slice of size `window`, that has x at its center.

        Note, that calculating the residues tends to be quite costy, because a function
        fitting is performed for every sample. To improve performance, consider the
        following possibilities:

        In case your data is sampled at an equidistant frequency grid:

        (1) If you know your data to have no significant number of missing values,
        or if you do not want to calculate residues for windows containing missing values
        any way, performance can be increased by setting min_periods=window.

        Note, that the initial and final window/2 values do not get fitted.

        Each residual gets assigned the worst flag present in the interval of
        the original data.

        Parameters
        ----------
        field : str
            The column, holding the data-to-be-modelled.

        window : {str, int}
            The size of the window you want to use for fitting. If an integer is passed,
            the size refers to the number of periods for every fitting window. If an
            offset string is passed, the size refers to the total temporal extension. The
            window will be centered around the vaule-to-be-fitted. For regularly sampled
            timeseries the period number will be casted down to an odd number if even.

        order : int
            The degree of the polynomial used for fitting

        min_periods : int or None, default 0
            The minimum number of periods, that has to be available in every values
            fitting surrounding for the polynomial fit to be performed. If there are not
            enough values, np.nan gets assigned. Default (0) results in fitting
            regardless of the number of values present (results in overfitting for too
            sparse intervals). To automatically set the minimum number of periods to the
            number of values in an offset defined window size, pass np.nan.
        """
        pass

    def calculateRollingResidues(self, field, window, func, min_periods, center):
        """
        Calculate the diff of a rolling-window function and the data.

        Note, that the data gets assigned the worst flag present in the original data.

        Parameters
        ----------
        field : str
            The column to calculate on.
        window : {int, str}
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.
        func : Callable, default np.mean
            Function to roll with.
        min_periods : int, default 0
            The minimum number of periods to get a valid value
        center : bool, default True
            If True, center the rolling window.
        """
        pass

    def roll(self, field, window, func, min_periods, center):
        """
        Calculate a rolling-window function on the data.

        Note, that the data gets assigned the worst flag present in the original data.

        Parameters
        ----------
        field : str
            The column to calculate on.
        window : {int, str}
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string
            is passed, the size refers to the total temporal extension. For regularly
            sampled timeseries, the period number will be casted down to an odd number if
            ``center=True``.
        func : Callable, default np.mean
            Function to roll with.
        min_periods : int, default 0
            The minimum number of periods to get a valid value
        center : bool, default True
            If True, center the rolling window.
        """
        pass

    def assignKNNScore(
        self, field, target, n, func, freq, min_periods, method, metric, p
    ):
        """
        TODO: docstring need a rework
        Score datapoints by an aggregation of the dictances to their k nearest neighbors.

        The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).

        The steps taken to calculate the scores are as follows:

        1. All the timeseries, given through ``field``, are combined to one feature space by an *inner* join on their
           date time indexes. thus, only samples, that share timestamps across all ``field`` will be included in the
           feature space.
        2. Any datapoint/sample, where one ore more of the features is invalid (=np.nan) will get excluded.
        3. For every data point, the distance to its `n` nearest neighbors is calculated by applying the
           metric `metric` at grade `p` onto the feature space. The defaults lead to the euclidian to be applied.
           If `radius` is not None, it sets the upper bound of distance for a neighbor to be considered one of the
           `n` nearest neighbors. Furthermore, the `freq` argument determines wich samples can be
           included into a datapoints nearest neighbors list, by segmenting the data into chunks of specified temporal
           extension and feeding that chunks to the kNN algorithm seperatly.
        4. For every datapoint, the calculated nearest neighbors distances get aggregated to a score, by the function
           passed to `func`. The default, ``sum`` obviously just sums up the distances.
        5. The resulting timeseries of scores gets assigned to the field target.

        Parameters
        ----------
        field : list of str
            input variable names.
        target : str, default "kNNscores"
            A new Column name, where the result is stored.
        n : int, default 10
            The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
        func : Callable[numpy.array, float], default np.sum
            A function that assigns a score to every one dimensional array, containing the distances
            to every datapoints `n` nearest neighbors.
        freq : {np.inf, float, str}, default np.inf
            Determines the segmentation of the data into partitions, the kNN algorithm is
            applied onto individually.

            * ``np.inf``: Apply Scoring on whole data set at once
            * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
            * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
              string

        min_periods : int, default 2
            The minimum number of periods that have to be present in a partition for the kNN scoring
            to be applied. If the number of periods present is below `min_periods`, the score for the
            datapoints in that partition will be np.nan.
        method : {'ball_tree', 'kd_tree', 'brute', 'auto'}, default 'ball_tree'
            The search algorithm to find each datapoints k nearest neighbors.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.
        metric : str, default 'minkowski'
            The metric the distances to any datapoints neighbors is computed with. The default of `metric`
            together with the default of `p` result in the euclidian to be applied.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.
        p : int, default 2
            The grade of the metrice specified by parameter `metric`.
            The keyword just gets passed on to the underlying sklearn method.
            See reference [1] for more information on the algorithm.

        References
        ----------
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        pass

    def copyField(self, field):
        """
        Copy data and flags to a new name (preserve flags history).

        Parameters
        ----------
        field : str
            The fieldname of the data column, you want to fork (copy).
        """
        pass

    def dropField(self, field):
        """
        Drops field from the data and flags.

        Parameters
        ----------
        field : str
            The fieldname of the data column, you want to drop.
        """
        pass

    def renameField(self, field, new_name):
        """
        Rename field in data and flags.

        Parameters
        ----------
        field : str
            The fieldname of the data column, you want to rename.
        new_name : str
            String, field is to be replaced with.
        """
        pass

    def maskTime(self, field, mode, mask_field, start, end, closed):
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
        field : str
            The fieldname of the column, holding the data-to-be-masked.
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
        pass

    def plot(
        self, field, path, max_gap, stats, history, xscope, phaseplot, store_kwargs
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
        field : str
            Name of the variable-to-plot

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

        stats : bool, default False
            Whether to include statistics table in plot.

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

        stats_dict: dict, default None
            (Only relevant if ``stats = True``)
            Dictionary of additional statisticts to write to the statistics table
            accompanying the data plot. An entry to the stats_dict has to be of the form:

            * ``{"stat_name": lambda x, y, z: func(x, y, z)}``

            The lambda args ``x``,``y``,``z`` will be fed by:

            * ``x``: the data (``data[field]``).
            * ``y``: the flags (``flags[field]``).
            * ``z``: The passed flags level (``kwargs[flag]``)

            See examples section for examples

        Examples
        --------
        Summary statistic function examples:

        >>> func = lambda x, y, z: len(x)

        Total number of nan-values:

        >>> func = lambda x, y, z: x.isna().sum()

        Percentage of values, flagged greater than passed flag (always round float results
        to avoid table cell overflow):

        >>> func = lambda x, y, z: round((x.isna().sum()) / len(x), 2)
        """
        pass

    def transform(self, field, func, freq):
        """
        Function to transform data columns with a transformation that maps series onto series of the same length.

        Note, that flags get preserved.

        Parameters
        ----------
        field : str
            The fieldname of the column, holding the data-to-be-transformed.
        func : Callable[{pd.Series, np.array}, np.array]
            Function to transform data[field] with.
        freq : {None, float, str}, default None
            Determines the segmentation of the data into partitions, the transformation is applied on individually

            * ``np.inf``: Apply transformation on whole data set at once
            * ``x`` > 0 : Apply transformation on successive data chunks of periods length ``x``
            * Offset String : Apply transformation on successive partitions of temporal extension matching the passed offset
              string
        """
        pass


class SaQCResult:
    def __init__(
        self,
        data: DictOfSeries,
        flags: Flags,
        attrs: dict,
        scheme: TranslationScheme,
    ):
        assert isinstance(data, DictOfSeries)
        assert isinstance(flags, Flags)
        assert isinstance(attrs, dict)
        assert isinstance(scheme, TranslationScheme)
        self._data = data.copy()
        self._flags = flags.copy()
        self._attrs = attrs.copy()
        self._scheme = scheme
        self._validate()

        try:
            self._scheme.backward(self._flags, attrs=self._attrs)
        except Exception as e:
            raise RuntimeError("Translation of flags failed") from e

    def _validate(self):
        if not self._data.columns.equals(self._flags.columns):
            raise AssertionError(
                "Consistency broken. data and flags have not the same columns"
            )

    @property
    def data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._data.copy().to_df()
        data.attrs = self._attrs.copy()
        return data

    @property
    def flags(self) -> pd.DataFrame:
        data: pd.DataFrame = self._scheme.backward(self._flags, attrs=self._attrs)
        data.attrs = self._attrs.copy()
        return data

    @property
    def data_raw(self) -> DictOfSeries:
        return self._data

    @property
    def flags_raw(self) -> Flags:
        return self._flags

    @property
    def columns(self) -> DictOfSeries():
        self._validate()
        return self._data.columns

    def __getitem__(self, key):
        self._validate()
        if key not in self.columns:
            raise KeyError(key)
        data_series = self._data[key].copy()
        # slice flags to one column
        flags = Flags({key: self._flags._data[key]}, copy=True)

        df = self._scheme.backward(flags, attrs=self._attrs)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=0, axis=1)

        if len(df.columns) == 1:
            df.columns = ["flags"]

        df.insert(0, column="data", value=data_series)
        df.columns.name = None
        df.index.name = None
        return df

    def __repr__(self):
        return f"SaQCResult\nColumns: {self.columns.to_list()}"
