#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
import functools
import inspect
import uuid
from abc import ABC, abstractmethod
from functools import cache
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.core.variable import Choice, Integer, Real
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from saqc.funcs.scores import _density
from saqc.lib.tools import getFreqDelta, timedeltaToOffset, toSequence

ANOMALY_ALIAS = dict(
    # FitUniLOF=["outlier", "outliers", "spike", "spikes"], # UniLOF problem overfits in current configuration.
    FitZScorePipeline=["outlier", "outliers", "spike", "spikes"],
    FitZScore=["outlierz", "outlierz", "spikez"],
    FitConstants=["constants", "constant"],
    FitScatterLowpass=["noise"],
    LowpassResids=["filter"],
    FitRange=["range"],
    FitGeneralPipeline=["general"],
)

PROBLEM_ALIAS = dict({a: p for p in ANOMALY_ALIAS.keys() for a in ANOMALY_ALIAS[p]})


class SaQCProblem(ABC, ElementwiseProblem):
    """
    Abstract Base Class for definitions of saqc-Problems.

    Notes
    -----
    Sub classes need to implement following methods
        * ``paraIni``: Assigns the variable types and ranges the pipeline is parametrised with
        * ``funcBody``: Holds the Function representing the Pipeline
        * ``objective``: Assigns cost function to minimize and the constraints alignment

    Data and target flags
        * The dataset and target flags are set during the initialisation of an actual problem
        * see ``__init__`` documentation

    Optimisation
        * Problems are intended to be solved/ran via ``pymoo.minimize`` function
        * During optimisation, ``_evaluate`` is called for evaluating parameter sets in a given iteration.
        * ``paraIni``, ``funcBody`` and ``objective`` are called within ``_evaluate``.

    See Also
    --------
    * :py:meth:`~saqc.lib.problems.SaQCProblem
    .paraIni`
    * :py:meth:`~saqc.lib.problems.SaQCProblem
    .funcBody`
    * :py:meth:`~saqc.lib.problems.SaQCProblem
    .objective`
    * :py:meth:`~saqc.lib.problems.SaQCProblem
    ._evaluate`

    """

    @property
    def used_names(self):
        """
        Property to keep track of names already used (for fields in config composition)
        """
        return tuple(self._used_names)

    @used_names.setter
    def used_names(self, value: str | tuple[str] | list[str]):
        """
        Assigning to used names adds the elements to the set of used names.
        """
        if isinstance(value, str):
            value = (value,)
        if isinstance(value, list | tuple):
            for e in value:
                self._used_names.add(e)
        else:
            raise ValueError(
                f"Can only Add strings, tuples or lists of strings to used_names. Got {value}"
            )

    @used_names.deleter
    def used_names(self):
        """
        Deletion resets the set of used names
        """
        self._used_names = set()

    @property
    def _tmp_suffix(self):
        """return a suffix string used as attachment to temporalily generated fields"""
        return self._tmp_suff

    @_tmp_suffix.setter
    def _tmp_suffix(self, value: str | int):
        """Set the basic string of the suffix string and set the counter to ``0`` (str) or increase the counter with
        the current suffix stringone (int)"""

        if isinstance(value, str):
            self._tmp_suff = "_" + value + "_0"
        elif isinstance(value, int):
            _split = self._tmp_suff.split("_")
            _split[-1] = str(int(_split[-1]) + value)
            self._tmp_suff = "_".join(_split)
        else:
            raise ValueError(
                f"Value for setting suffix for temporalily gnenerated strings, values must be string or Integer. Got {value}"
            )

    def getTmpName(self, name: str, ex_fields: tuple[str] = tuple()):
        """Get a new name string, that is composed of ``name`` and a suffix string. Increase the counter for the
        suffix string as long as the result is an element of ``ex`` or of ``self.used_names``.
        """
        while True:
            out = name + self._tmp_suffix
            self._tmp_suffix = 1
            if (out not in ex_fields) and (out not in self._used_names):
                break
        self.used_names = out
        return out

    def _configFunc(self, type: str = "field"):
        """Generates a function, that, when called at the stage of config composing, yields a value as intended
        during problem wise config definition"""

        if type == "field":
            # make a function that will just return "var"
            return lambda var, ex, X: var
        if type == "new_field":
            # make a function that will return a new field name upon first call, and than always return that same
            # generated new Field on subsequent calls.
            cf = cache(self.getTmpName)

            def newField(var, ex, X, cf=cf):
                return cf(var, ex)

            return newField
        else:
            # make a function that returns the evaluation of parameter ``type`` at X
            def paraEval(var, ex, X, type=type):
                return self.conversions(X)[type]

            return paraEval

    @property
    def config(self):
        """
        Holds a list of tuples that represent config lines
        """
        return self._config or []

    @config.setter
    def config(self, value):
        if not isinstance(value, list):
            value = [value]
        self._config = value

    def getFixedConfig(
        self, var: str = None, ex: tuple = None, X: dict = None, label: str = None
    ):
        """
        Fixate the config lines represented by ``self.config`` by calling all callables in the signature dictionaries.
        """
        if var is None:
            var = self.var
        if ex is None:
            ex = tuple(self.qc.columns)
        if X is None:
            X = {}
        if label is None:
            label = {}
        else:
            label = {"label": label}

        fix_conf = self.config.copy()
        for c in fix_conf:
            # fixate the signature by calling the Callables it holds
            for k, v in c[1].items():
                if isinstance(v, Callable):
                    c[1][k] = v(var, ex, X)
            # add label to signature
            c[1].update(label)
            # check for parameters that have been set implicitly during optimisation (defaults)
            sig = inspect.signature(getattr(self.qc, c[0]))
            kwarg_params = [
                (k, v.default)
                for k, v in sig.parameters.items()
                if v.default != inspect.Parameter.empty
            ]
            # add the defaults to config
            for kwarg in kwarg_params:
                if (kwarg[0] not in c[1].keys()) and (
                    kwarg[0] not in ["dfilter", "flag"]
                ):
                    c[1].update({kwarg[0]: kwarg[1]})

        if len(fix_conf) == 1:  # for single row configs, add the optimised parameters
            fix_conf[0][1].update({p: self.conversions(X)[p] for p in self.parameters})

        return fix_conf

    def getBias(self) -> list[dict]:
        """
        Return parameter population sampled according to specifications made via ``self.setBias``.
        Parameters with no biased sampling assigned will be sampled uniformly from the search space.
        """

        if self._bias is None:
            dummy = lambda: {
                p: self.parameters[p].sample(1)[0] for p in self.parameters
            }
            self.setBias(dummy, bias_portion=0)
        return self._bias

    def setBias(
        self,
        value: list[dict] | Callable,
        mode: Literal["reset", "prepend"] = "reset",
        bias_portion=0.5,
    ):
        """
        Set bias for initial sampling.

        Only ``bias_portion`` of the population will come from the bias, where the
        remainder is populated with uniform sampling.
        Parameters, that are not determined by the bias definition, will
        be uniformly sampled along with the biased parameters.

        Parameters
        ----------
        value :
            If ``Callable``: use it to sample individuals until population size is reached
            If ``list``: assign listed items as bias

        mode :

        bias_portion :

        Notes
        -----
        * Population is expected to be examplified as sequence of dictionaries
          with the problems parameter names as keys and evaluations as values.
        * Unbiased parameters will be filled with uniform sampling.
        * All individuals as by `value`` have to follow paramer bounds of the problem.

        Example
        -------

        Assuming a problem with parameters ``"a"`` and ``"b"`` and a bias with 3 individuals, list defined bias
        Assuming a problem with parameters ``"a"`` and ``"b"`` and a bias with
        3 individuals, list defined bias would have to look as follows:

        .. doctest::

           >>> value = [{"a":a_value_0, "b": b_value_0}, {"a":a_value_1, "b": b_value_1}, {"a":a_value_2, "b": b_value_2}]

        If bias should only be applied to ``"a"``, where ``"b"`` is just to be
        sampled uniformly from search space, ``"b"`` can just be dropped from
        the individual dicts:

        .. doctest::

           >>> value = [{"a":a_value_0}, {"a":a_value_1}, {"a":a_value_2}]

        All individuals have to be composed of the same parameters. The following will NOT work:

        .. doctest::

           >>> value = [{"a":a_value_0}, {"a":a_value_1, "b": b_value_1}, {"a":a_value_2}]

        Value can also just be a function jielding one (randomly sampled) individual at a time:

        .. doctest::
           >> import numpy as np
           >> value = lambda: {"a": np.random.normal()*5, "b": np.random.normal() + 10}

        All parameters of individuals given by `value`` have to follow
        parameter bounds of the problem.
        """

        pop_size = self.algorithm_kwargs["pop_size"]
        bias_size = int(np.max([1.0, np.round(pop_size * bias_portion)]))
        if not isinstance(value, list):
            _value = lambda k: value()
        else:
            if len(value) == 0:
                return None
            bias_size = np.min([bias_size, len(value)])
            _value = lambda k, l=value: l[k]
        uniform_size = pop_size - bias_size
        uniform_paras = [p for p in self.parameters if p not in _value(0)]
        uni_df = pd.DataFrame(columns=uniform_paras)
        for u in uniform_paras:
            uni_df[u] = self.parameters[u].sample(bias_size)
        biased = [
            dict(_value(k), **{u: uni_df[u][k] for u in uniform_paras})
            for k in range(bias_size)
        ]
        if mode in ["prepend", "append"]:
            bias = self.getBias()
            bias = biased[:bias_size] + bias[: pop_size - bias_size]
        else:  # mode == "reset"
            uni_df = pd.DataFrame(columns=list(self.parameters.keys()))
            for u in self.parameters:
                uni_df[u] = self.parameters[u].sample(uniform_size)
            bias = biased + [
                {u: uni_df[u][k] for u in self.parameters} for k in range(uniform_size)
            ]

        self._bias = bias

    def optimizerIni(self) -> dict:
        """
        Returns dictionary holding parameterisation of `pymoo.minimize` with
        instantiated algorithm object as determined by ``self``.
        Especially it assigns (possibly biased) initialisation samples to the algorithm instance.
        """

        b_pop = Population.new("X", self.getBias())
        Evaluator().eval(self, b_pop)
        algorithm_kwargs = dict(self.algorithm_kwargs, **dict(sampling=b_pop))
        out = dict(
            self.optimizer_kwargs,
            **{"algorithm": self.optimizer_kwargs["algorithm"](**algorithm_kwargs)},
        )
        return out

    def __init__(
        self,
        qc: "SaQC",
        target_flags: pd.Series[bool],
        skip_mask: pd.Series[bool] = None,
        optimizer_kwargs: dict = {},
        algorithm_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initialise Problem instance by assigning data, target and optimizer context.

        qc :
            Univariate, unflagged ``SaQC`` object holding the data, the optimisation will be performed on.

        target_flags :
            Series evaluating ``True``, where optimised saqc pipeline is expected to set flags.

        skip_mask :
            Series evaluating ``True`` where results from the optimised flagging pipeline shall not
            be included into the scoring.
            Excluded indices will neither contribute to FalsePositives nor to FalseNegatives count.

        optimizer_kwargs :
            Configuration of the optimizer used. Will override configurations implied in class definition
            Can especially be used to set termination requirements of the run via passing ``{"termination": ("n_evals", n_values)}``

        algorithm_kwargs :
            Configuration of the algorithm used. Will override configurations implied in class definition
            Can especially be used to set termination requirements of the run via passing ``{"pop_size": n_individuals}``
        """

        # init problem type (does the problem change the data or the flags: defaults to flags)
        self.processing_type: Literal["flags", "data"] = "flags"

        # init initialisation bias (no bias)
        self._bias = None

        # is problem instantiated as part of a problem sequence ?
        self._is_merged = kwargs.get("_is_merged", 0)

        # config related attributes
        self._config = None
        self._used_names = set()
        self._tmp_suffix = "_TEMP"

        # init optimizer keywords dictionary
        self.optimizer_kwargs: dict = {
            "algorithm": MixedVariableGA,
            "survival": RankAndCrowding(crowding_func="mnn"),
            **optimizer_kwargs,
        }

        # init algorithm kwargs with user spec (or empty dict)
        self.algorithm_kwargs = algorithm_kwargs

        # Design variables and their bounds - needs be assigned in subclasses ``paraIni``.
        self.parameters: dict = NotImplementedError

        # Conversion Function default.
        # can be adjusted in subclasses in case optimized variables are not identical with saqc parameters
        self.conversions = lambda x: x

        # multivariate direction of "less" strict flagging overall
        self.isolation_directions: list = []

        # multivariate direction of "stricter" flagging
        self.generalisation_directions: list = []

        # assign qc object (holding data)
        self.qc = qc

        # for conveniance, memorize only the field name in ``self.qc``
        self.var = qc.data.columns[0]

        # assign targeted flags
        self.target_flags = target_flags

        # assign skip mask, values at skipmasked periods are still fed to
        # the flags calculation algorithm - but do not contribute to scoring
        self.skip_mask = self.qc.data[
            self.var
        ].isna()  # defaultly assigning NaN evaluations
        if skip_mask is not None:
            self.skip_mask |= skip_mask  # join passed skipmask

        # initialise ``._flags`` , this attribute will be updated with every problem iterations flagging result
        self._flags = self.target_flags & (~self.skip_mask)

        # assign confusion matrix scores
        self._confusion()  # initialise confusion from flagging

        self.mode = int(not self._flags.any())

        # initalise Parameter types, ranges, and other characteristics
        self.paraIni()

        # assign transformation operator used for scaling and offsetting of parameters
        self.T = functools.partial(self._T, self.parameters)

        # init cache function

        self._cache = self._processing_cache()

        # derive generalisation directions from isolation directions
        self.generalisation_directions = [
            ("~" + v) if (v[0] != "~") else v[1:] for v in self.isolation_directions
        ]

        # derive number of objectives and constraints the instantiated problem comes with
        n_obj, n_ieq_constr = self._getObjectiveShape()

        # init pymoo parent
        super().__init__(
            vars=self.parameters, n_obj=n_obj, n_ieq_constr=n_ieq_constr, **kwargs
        )

    def conTuner(self, res):
        """
        Yield a new problem instance of same type as ``self``, but operating in
        generalisation mode, confined to a subspace of the searchspace, where the
        flagging result (as given by ``self.falsification_count()`` doesn't get
        worse than that obtained from evaluating ``res``.
        """
        _obj = {}
        self.mode = 1
        self.n_ieq_constr += 1
        self._evaluate(res.X, _obj)
        self.target_flags = self.TP  # correct detections
        self.skip_mask = self.FN | self.FP | self.skip_mask  # false detections
        falsification_pop = res.pop.get("F").astype(int)
        falsification_pop = falsification_pop == falsification_pop.min()
        solutions = list(res.pop.get("X")[falsification_pop.squeeze()])
        self.setBias(solutions, mode="prepend", bias_portion=0.1)

        return self

    @abstractmethod
    def paraIni(self) -> dict:
        """
        Initialise types, ranges, conversions and transformations of the parameters
        that are to be optimized.

        Method is supposed to be overridden in child class definitions.

        Following assignments are to be done in method body:

        * ``self.parameters``: Dictionary with parameter names as keys and
          ``pymoo.core.variable`` instances as values.
        * ``self.conversion``: A function modifying and returning parameter
            dictionary, the problem is called with.
        * ``self.bias``: See property docstring of ``bias``
        * ``self.config``: List of tuples representing config file lines of the problem
        """

    @staticmethod
    @abstractmethod
    def funcBody(qc: "SaQC", X: dict, **kwargs: dict) -> "SaqC":
        """
        Computational body of the problem.

        Method is called to evaluate the *SaQC* pipeline under the parametrisation currently
        under Test (``X``).

        Method is supposed to be overridden in child class definitions.


        Parameters
        ----------

        qc :
           SaQC instance holding the data the pipeline should be optimized on.

        X :
           Dictionary containing target parameter evaluations as values and
           target parameter names as keys

        kwargs :
           Additional parameters to be passed on to the ``saqc`` methods wrapped.
           This is useful for passing pipeline controlling parameters (such as `flag` or
           `label`), as well as alternations to method default parameters that weren't
           optimimized when calling the SaQC method obtained from the optimized pipeline.


        Returns
        -------
        qc_out :
            return the input ``qc`` object with the calculated flags assigned to ``field``.

        """
        pass

    @abstractmethod
    def objective(self, X: dict, out: dict):
        """
        Objective Function of the Problem.

        Method is called during any minimisation-iteration to update objective
        scores and constraints alignment (``out``) for the parameter set (``X``)
        currently under test.

        Method is supposed to be overridden by child classes.

        Parameters
        ----------
        X :
           Dictionary containing target parameter evaluations as values and
           target parameter names as keys

        out :
            Dictionary where objective Function evaluations should be written to/updated
            inplace:
            out['F'] - Evaluation Score for the current parameter Set.
            out['G'] - Constraints evaluation: passing ``out['G'] = f(X)`` ensures f(X) <= 0

        Notes
        -----
        In the function body, updated confusion attributes can be accessed via:

            * ``self.FP`` - false positives
            * ``self.FN`` - false negatives
            * ``self.TP`` - true positives
            * ``self.TN`` - true negatives

        In the function body, the class built in scoring functions can be accessed.

            * ``self.falsification_count`` - Additive penalizing of `FP` and `FN`.
            * ``self.fscore`` - generic *Fscore* wrapper

        Also attributes assigned in ``self.paraIni`` can be used for penalizing.
        """
        pass

    def _getObjectiveShape(self):
        """
        Helper function to derive the number of objectives and constraints
        upon instantiation generically
        """

        _objective = {}
        _X = {var: self.parameters[var].sample(1)[0] for var in self.parameters.keys()}
        self.objective(_X, _objective)
        n_obj = len(toSequence(_objective["F"]))
        n_ieq_constr = len(toSequence(_objective.get("G", [])))
        return n_obj, n_ieq_constr

    @staticmethod
    def _T(
        parameters: dict, X: dict, key: str, scale: float = None, base: float = None
    ):
        """
        Linear fitness derived from design variables:

        maps abs(X[key] - base) to [0,scale] equidistant. Origins of 0 and scale
        are derived from the parameter bounds.

        """
        inv, key = (True, key[1:]) if key[0] == "~" else (False, key)
        x, l, u = X[key], parameters[key].bounds[0], parameters[key].bounds[1]
        b = base if base is not None else l
        if (b < l) | (b > u):
            ValueError(f"Base value {b} not within bounds {l} (lower) and {u} (upper)")
        x, l, u, b = x - l, 0, np.max([u - b, b - l]), b - l
        if scale is not None:
            x, u, b = (x * scale / u), scale, (b * scale / u)
        x = abs(x - b)
        x = u - x if inv else x
        return x

    def _confusion(self):
        """
        Calculates the confusion matrix in any iteration by assigning to the
        confusion attributes of the problem
        """

        self.FP = self._flags & ~self.target_flags & ~self.skip_mask  # .sum()
        self.FN = self.target_flags & ~self._flags & ~self.skip_mask  # .sum()
        self.TP = self._flags & self.target_flags & ~self.skip_mask  # .sum()
        self.TN = ~self.target_flags & ~self._flags & ~self.skip_mask  # .sum()

    def falsification_count(
        self,
    ) -> float:
        """
        Calculate additive cost function.

        With FP = False Positives, FN = False Negatives, the following is calculated:
        cost = FP + FN
        """
        return self.FP.sum() + self.FN.sum()

    def f_score(self, beta: float = 1, pen_add: float = 0, pen_mul: float = 1) -> float:
        """
        Calculate *Fscore* cost Function

        Calculates the usual (beta dependend) FScore from the current confusion matrix evaluations
        and derives the penalised cost function as follows:

        cost = (1 - FScore)*pen_mul + pen_add
        """
        b = beta**2
        f = (
            (1 + b)
            * self.TP.sum()
            / (((1 + b) * self.TP.sum()) + self.FP.sum() + b * self.FN.sum())
        )
        return (1 - f) * pen_mul + pen_add

    def _evaluate(self, X: dict, out: dict, *args, **kwargs):
        """
        Function called every iteration to evaluate the pipeline.

        Evaluates the pipeline under the currently tested parameter set (``X``)
        and assigns/updates the current cost-scores and constraints evaluations in ``out``.

        Notes
        -----
        Following steps are performed (in that order)

        0. Parameterset ``X`` is converted to SaQC - readable format
        1. Call to ``funcBody()`` to get flagging results for Parameterset currently under test
        2. Derive Confusion Attributes from the flagging result obtained in 1.
        3. Call to ``objective`` to calculate and assign objective function evaluation and
           constraints Alignment for the currently tested Parameterset
        """

        X_conv = self.conversions(X)
        self._flags = (
            self.funcBody(
                self.qc, self.var, X_conv, _processing_cache=self._cache
            ).flags[self.var]
            > -np.inf
        )
        self._confusion()
        self.objective(X, out)
        return out

    def _evalAt(self, X, do_plot=True):
        """
        For debugging purposes mainly:
        Evaluate problem at individual X and
        return an SaQC Object with fields:

        * Flags (data with flags according to flagging outcome under X)
        * Targets (data with flags according to Targets of X)
        * Skipped (data with flags where problem ignores score contributions)
        """
        _out = {}
        self._evaluate(X, _out)
        _qc = self.qc.copy()

        fields = ["Flags", "Targets", "Skipped"]
        property = ["_flags", "target_flags", "skip_mask"]
        for f, p in zip(fields, property):
            _qc = _qc.copyField(self.var, f)
            _qc = _qc.flagGeneric(f, lambda x: getattr(self, p))

        if do_plot:
            _qc.plot(fields, ax_kwargs={"ncols": 1}, mode="subplots")
        return _qc

    def _processing_cache(self):
        """If set, should return a cached function"""
        pass

    def problemInfo(self):
        """
        Return print ready string, specifying problem type and parameters in a readable way.
        """
        tab = " " * 4
        problem_name = self.name()
        parameters = [
            f"{p[0]}: {p[1].vtype.__name__} in {p[1].bounds if hasattr(p[1],'bounds') else p[1].options}"
            for p in self.parameters.items()
        ]
        parameters = [f"{tab}{p}" for p in parameters]

        problem_str = f"Problem: {problem_name}\n"
        parameter_str = f"Parameters: \n"
        parameter_list_str = "\n".join(parameters) + "\n"
        pop_str = f'{tab}population: {self.algorithm_kwargs["pop_size"]}\n'
        solve_str = f"Solver:\n"
        alg_str = f'{tab}algorithm:{self.optimizer_kwargs["algorithm"].__name__}\n'
        termination = (
            f'{tab}termination: {self.optimizer_kwargs.get("termination", "auto")}\n'
        )
        print_str = f" {problem_str}\n {parameter_str}{parameter_list_str}\n"
        print_str += f" {solve_str}{alg_str}{pop_str}{termination}"
        return print_str

    def switchMode(
        self,
        res,
    ):
        """
        Changes the "mode" of the Problem from 0 ("False Negative Direction") to 1 ("False Positive Direction")
        In ``mode = 0`` -> ``False Negative Direction``, the Problem target is

        1. To minimize the False Detections (FP + FN)
        2. Among all solutions that minimize the False Detection, To find the one that is the "closest" to a solution that would
            produce one more False Negative.

        In ``mode = 1`` -> ``False Positive Directionn``, the Problem target is
        1. Among all solutions that yield the same flagging result as the one found above,
           To find the one that is the "closest" to a solution that would
           produce one more False Positive.
        """
        # new mode
        self.mode = 1
        # in new mode Problem has the extra constraint to the Subset of optimal solutions
        self.n_ieq_constr += 1
        self._evaluate(
            res.X, {}
        )  # set the Problem to the Solution with the new optimal level
        self.target_flags = (
            self.TP
        )  # The target detections are exactly those produced with the optimal solution
        self.skip_mask = (
            self.FN | self.FP | self.skip_mask
        )  # Do not factor into the score the False detections the optimal
        # solution produced
        falsification_pop = res.pop.get("F").astype(int)
        # Get the Objective function values of the converged population
        falsification_pop = falsification_pop == falsification_pop.min()
        # find all individual indices yielding the optimal falsification count
        solutions = list(res.pop.get("X")[falsification_pop.squeeze()])
        # make sure bias contains current Optimum:
        solutions[0] = res.X
        # add those to bias of problem in new mode
        self.setBias(solutions, mode="prepend", bias_portion=0.1)


class FitUniLOF(SaQCProblem):
    """
    The Problem of optimizing ``flagUniLOF(n, thresh)``
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """
        d = self.qc.data[self.var]
        # use na filling for _density only if the data diff is too sparse
        fill_na = len(d) / (d.diff().isna().sum()) < 2
        # retrieve result from automatic density calculation happening in flgUniLOF to fixate it
        # (otherwise uniLOF will not generalize well)
        density, *_ = _density(d, p=1, fill_na=fill_na, density="auto")
        self.parameters = dict(
            n=Integer(bounds=(2, 50)),
            thresh=Real(bounds=(1, 100)),
            density=Choice(
                value=density, options=[density]
            ),  # dummy choice to fixate density at calibration
        )

        def bias():
            n = np.random.randint(10, 30)
            thresh = np.random.random() * 20 + 1
            return dict(n=n, thresh=thresh)

        self.setBias(bias)
        self.config = (
            "flagUniLOF",
            dict(slope_correct=True, min_offset=None, algorithm="kd_tree", p=1),
        )

    def _processing_cache(self):
        @cache
        def _cache_func(n: int, density: float):
            return self.qc.assignUniLOF(
                self.var,
                n=n,
                statistical_extent=1,
                algorithm="kd_tree",
                density=density,
                p=1,
            ).data[self.var]

        return _cache_func

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """
        tmp_field = "VAR" + str(uuid.uuid4()).replace("-", "_")
        cached: dict | None = kwargs.get("_processing_cache", None)
        if cached is not None:

            _qc = qc.copy()
            _qc[tmp_field] = type(qc)(cached(X["n"], X["density"]).rename(tmp_field))
            # print(f"n-value: {X['n']}")
            # print(cached.cache_info())
        else:
            _qc = qc.assignUniLOF(
                field,
                target=tmp_field,
                n=X["n"],
                density=X["density"],
                statistical_extent=1,
                algorithm="kd_tree",
                p=1,
                **kwargs,
            )
        para_check = 2 if X["thresh"] is not None else 3
        _kwargs = dict(
            slope_correct=True,
            min_offset=None,
            density=X["density"],
            flag=255.0,
            para_check=para_check,
            thresh=X["thresh"],
            corruption=None,
            probability=None,
        )
        return _qc._uniLOF(
            field, n=X["n"], tmp_field=tmp_field, **dict(_kwargs, **kwargs)
        ).dropField(tmp_field)

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """

        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                # + self.T(X, "~probability", 0.5, 1)
                + self.T(X, "~thresh", 1)
                # + self.T(X, "min_offset", 1)
            ]
        elif self.mode == 1:
            out["F"] = [self.T(X, "thresh", 1)]  # + self.T(X, "min_offset", 1)]
            # out["F"] = [self.T(X, "probability", 1)]
            out["G"] = [self.falsification_count()]
        else:
            raise ValueError(f"Optimisation Mode {self.mode} not known.")


class FitScatterLowpass(SaQCProblem):
    """
    The Problem of optimizing ``flagScatterLowpass(window, sub_window,thresh,sub_thresh)``
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """

        d = self.qc.data[self.var]
        d[self.skip_mask]
        freq = pd.Timedelta(getFreqDelta(d.index))

        win_min = 2
        if self.mode == 0:
            f = self.target_flags | self.skip_mask
            groups = d.groupby((~f).cumsum().values)
            win_max = groups.count().max() * 1.25
            thresh_max = (
                ((0.5 * (groups.max() - groups.min()))).max() * win_min / (win_min - 1)
            )
        else:
            win_max = int(len(d))
            thresh_max = (
                pd.concat([d.nlargest(win_min), d.nsmallest(win_min)]).std() * 10 / 9
            )
        win_max = np.max([win_min + 1, win_max])
        self.parameters = dict(
            window=Integer(win_min + 1, bounds=(win_min, win_max)),
            thresh=Real(0, bounds=(0, thresh_max)),
            sub_window=Integer(win_min, bounds=(win_min, win_max)),
            sub_thresh=Real(0, bounds=(0, thresh_max)),
        )

        def conversion(X):
            x = X.copy()
            x["window"] = timedeltaToOffset(freq * X["window"])
            x["sub_window"] = timedeltaToOffset(freq * X["sub_window"])
            return x

        self.conversions = conversion
        self.isolation_directions = ["~window", "~thresh", "sub_window", "~sub_thresh"]

        pop_size = self.algorithm_kwargs["pop_size"]

        def bias():
            window = np.random.randint(win_min, np.min([pop_size * 2, win_max]))
            sub_window = np.random.randint(win_min, np.max([window, win_min + 1]))
            thresh = np.min([d.rolling(window).std().mean(), thresh_max])
            sub_thresh = np.min([d.rolling(sub_window).std().mean(), thresh])
            return dict(
                window=window,
                sub_window=sub_window,
                thresh=thresh,
                sub_thresh=sub_thresh,
            )

        self.setBias(bias)
        self.config = ("flagByScatterLowpass", dict())

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """
        return qc.flagByScatterLowpass(field, **X, **kwargs)

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """

        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                + sum(
                    [
                        self.T(X, d, (1 / len(self.isolation_directions)))
                        for d in self.isolation_directions
                    ]
                )
            ]
            out["G"] = [X["sub_window"] - X["window"], X["sub_thresh"] - X["thresh"]]
        else:
            out["F"] = [sum([self.T(X, d, 1) for d in self.generalisation_directions])]
            out["G"] = [
                self.falsification_count(),
                X["sub_window"] - X["window"],
                X["sub_thresh"] - X["thresh"],
            ]  # , X['sub_window']-X['window']]


class FitConstants(SaQCProblem):
    """
    The Problem of optimizing ``flagConstants(thresh, window)``
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """

        d = self.qc.data[self.var]
        freq = pd.Timedelta(getFreqDelta(d.index))
        if self.mode == 0:
            win_max = int(
                np.ceil(
                    self.target_flags.groupby((~self._flags).cumsum()).count().max()
                    * 1.1
                )
            )
        else:
            win_max = int(np.ceil(len(d) / 4))

        thresh_max = (
            d.max() - d.min()
        )  # d.diff().abs().unique()[1]  # d.max() - d.min()
        win_min = 2
        self.parameters = dict(
            # window=Integer(bounds=(2, win_max)),
            window=Integer(bounds=(np.min([win_min, win_max - 1]), win_max)),
            thresh=Real(bounds=(0, thresh_max)),
        )
        self.isolation_directions = ["~window", "thresh"]

        def conversion(X):
            x = X.copy()
            x["window"] = timedeltaToOffset(freq * X["window"])
            return x

        self.conversions = conversion

        def bias(win_min=win_min, win_max=win_max):
            window = np.random.randint(win_min, np.min([100, win_max]))
            _thresh_roll = d.rolling(window, min_periods=0)
            thresh = _thresh_roll.max() - _thresh_roll.min()
            thresh = abs(np.random.normal(thresh.mean(), thresh.std()))
            return {"window": window, "thresh": thresh}

        self.setBias(bias)
        # self.optimizer_kwargs.update(dict(termination=("n_evals",5000)))
        self.config = ("flagConstants", dict())

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """
        return qc.flagConstants(
            field,
            **X,  # freezing min_periods at the value of window to circumvent effects of a bug
            **kwargs,
        )

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """
        # out["F"] = self.falsification_count(pen_add=self._additivePenalty(**X))

        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                + sum(
                    [
                        self.T(X, d, (1 / len(self.isolation_directions)))
                        for d in self.isolation_directions
                    ]
                )
            ]
        elif self.mode == 1:
            # out["F"] = [self.T(X,'thresh',1) , self.T(X, 'window',1) , self.T(X, 'min_residuals',1)]
            out["F"] = [sum([self.T(X, d, 1) for d in self.generalisation_directions])]
            out["G"] = [self.falsification_count()]
        else:
            raise ValueError(f"Optimisation Mode {self.mode} not known.")


class FitRange(SaQCProblem):
    """
    The problem of optimising ``flagRange(min,max)``
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """

        d = self.qc.data[self.var]
        _min = d.min() - 1
        _max = d.max() + 1
        self.parameters = dict(
            min=Real(bounds=(_min, _max)),
            max=Real(bounds=(_min, _max)),
        )
        # try to choose "min" as high as possible and "max" as low as possible, (without messing up FP and FN)
        self.isolation_directions = ["min", "~max"]
        self.config = ("flagRange", dict())

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """
        return qc.flagRange(
            field,
            **X,
            **kwargs,
        )

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """
        # out["F"] = self.falsification_count(pen_add=self._additivePenalty(**X))

        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                + sum(
                    [
                        self.T(X, d, (1 / len(self.isolation_directions)))
                        for d in self.isolation_directions
                    ]
                )
            ]
            out["G"] = [X["min"] - X["max"]]  # -> "X["min"] - X["max"] < 0"
        elif self.mode == 1:
            # out["F"] = [self.T(X,'thresh',1) , self.T(X, 'window',1) , self.T(X, 'min_residuals',1)]
            out["F"] = [sum([self.T(X, d, 1) for d in self.generalisation_directions])]
            out["G"] = [self.falsification_count(), X["min"] - X["max"]]
        else:
            raise ValueError(f"Optimisation Mode {self.mode} not known.")


class SaQCProblemJoin(list):
    """
    Container class representing a join of Problems meant to jointly solve a common flagging target.
    Returns a MergedProblem instance composed of instances of the specified problems, when called.
    """

    def __call__(
        self,
        qc: "SaQC",
        target_flags: pd.Series[bool],
        skip_mask: pd.Series[bool] = None,
        optimizer_kwargs: dict = {},
        algorithm_kwargs: dict = {},
        **kwargs,
    ):
        call_kwargs = dict(
            qc=qc,
            target_flags=target_flags,
            skip_mask=skip_mask,
            optimizer_kwargs=optimizer_kwargs,
            algorithm_kwargs=algorithm_kwargs,
            **kwargs,
        )
        call_kwargs.pop("_is_merged", None)
        problem_i = [p(**dict(_is_merged=len(self), **call_kwargs)) for p in self]
        return merge_problems(problem_i)(**call_kwargs)


class LowpassResids(SaQCProblem):
    """
    Problem of fitting ``fitLowpassFilter(cutoff)``

    Processing Problem - does not contribute a meaningful objective function value by itself, but needs to be
    merged with a flagging problem.

    For example:

    -> ProblemJoin([FitButterHighPass,FitZScore]) Fits Z-Scoring to residuals obtained from Butterworth filter of data.
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """
        self.processing_type = "data"
        self.parameters = dict(
            cutoff=Integer(bounds=(1, 100)),
        )
        self.isolation_directions = ["cutoff"]

        def conversion(X):
            cutoff_func = lambda x: 1 - ((x + 0.000001) / (x + 1))
            x = X.copy()
            x["cutoff"] = (
                cutoff_func(X["cutoff"])
                if X["cutoff"] < self.parameters["cutoff"].bounds[-1]
                else None
            )
            x["cutoff_origin"] = X["cutoff"]  # need that for caching fitting results
            return x

        self.conversions = conversion

        # for config
        _target = self._configFunc("new_field")
        _field = self._configFunc("field")
        _fields = lambda var, ex, X: [_target(var, ex, {}), _field(var, ex, {})]
        _genFunc = (
            lambda var, ex, X: f"({_target(var, ex, {})} - {_field(var, ex, {})})"
        )
        self.config = (
            "fitLowpassFilter",
            dict(target=_target, cutoff=self._configFunc("cutoff")),
        )
        self.config += [
            ("processGeneric", dict(field=_fields, target=_field, func=_genFunc))
        ]
        self.config += [("dropField", dict(field=_target))]

    def _processing_cache(self):
        field = self.var
        target = field + "_trg"
        qc = self.qc.copy()
        dummy_dict = {p[0]: p[1].sample(1)[0] for p in self.parameters.items()}

        def cutoff_conversion(x):
            return self.conversions(dict(dummy_dict, **dict(cutoff=x)))["cutoff"]

        @functools.lru_cache(maxsize=100)
        def _cache_func(cutoff):
            qc_ = qc.fitLowpassFilter(
                field, target=target, cutoff=cutoff_conversion(cutoff)
            )
            return qc_.processGeneric(
                [field, target], target=target, func=lambda x, y: x - y
            ).data[target]

        return _cache_func

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """

        cached: dict | None = kwargs.get("_processing_cache", None)
        if X["cutoff"] is not None:
            if cached is not None:
                _qc = qc.copy()
                _qc[field] = type(qc)(cached(X["cutoff_origin"]).rename(field))
            else:
                tmp_field = "VAR" + str(uuid.uuid4()).replace("-", "_")
                _qc = qc.fitLowpassFilter(field, target=tmp_field, cutoff=X["cutoff"])
                _qc = _qc.processGeneric(
                    [field, tmp_field], target=field, func=lambda x, y: x - y
                ).dropField(tmp_field)
            return _qc
        else:
            return qc.processGeneric(field, func=lambda x: x - x.mean())

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """
        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                + sum(
                    [
                        self.T(X, d, (1 / len(self.isolation_directions)))
                        for d in self.isolation_directions
                    ]
                )
            ]
        elif self.mode == 1:
            # out["F"] = [self.T(X,'thresh',1) , self.T(X, 'window',1) , self.T(X, 'min_residuals',1)]
            out["F"] = [sum([self.T(X, d, 1) for d in self.generalisation_directions])]
            out["G"] = [self.falsification_count()]
        else:
            raise ValueError(f"Optimisation Mode {self.mode} not known.")


class FitZScore(SaQCProblem):
    """
    The problem of minimizing ``flagZScore``.
    Basic ZScoring Problem, with no data preprocessing included.
    """

    def paraIni(self) -> dict:
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .paraIni`
        """

        d = self.qc.data[self.var]
        d[self.skip_mask] = np.nan
        m = d.mean()
        m = np.max([d.max() - m, m - d.min()])
        thresh_max = 1000

        thresh_min = 0
        window_max = len(d)
        window_min = 5
        self.parameters = dict(
            thresh=Real(bounds=(thresh_min, thresh_max)),
            window=Integer(bounds=(window_min, window_max)),
            min_residuals=Real(bounds=(0, m)),
            method=Choice(options=["standard", "modified"]),
        )
        freq = pd.Timedelta(getFreqDelta(d.index))

        def conversion(X):
            x = X.copy()
            x["window"] = (
                timedeltaToOffset(freq * X["window"])
                if (X["window"] < window_max)
                else None
            )  # need that for caching fitting results
            return x

        self.conversions = conversion
        self.isolation_directions = [
            "~thresh",
            "~min_residuals",
            "~window",
        ]  # "~window", "~min_residuals"]

        win_bias = lambda: np.random.randint(
            np.min([window_min]), np.min([20 * window_min, window_max])
        )
        # cut_bias = lambda: np.random.randint(2,10,1)[0]
        thresh_bias = lambda: np.min(
            [np.max([np.random.standard_normal() + 3, 0.0]), thresh_max]
        )
        sample_func = lambda: {
            "window": win_bias(),
            "thresh": thresh_bias(),
        }
        self.setBias(sample_func)
        # prepend global ZScore trials (no rolling window)
        sample_func_global = lambda: {
            "window": window_min,
            "thresh": thresh_bias(),
        }
        self.setBias(sample_func_global, mode="prepend", bias_portion=0.1)
        self.config = ("flagZScore", dict())

    @staticmethod
    def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .funcBody`
        """
        return qc.flagZScore(field, **X, **kwargs)

    def objective(self, X, out):
        """
        See Also
        --------
        :py:meth:`~saqc.lib.problems.SaQCProblem
        .objective`
        """
        if self.mode == 0:
            out["F"] = [
                self.falsification_count()
                + sum(
                    [
                        self.T(X, d, (1 / len(self.isolation_directions)))
                        for d in self.isolation_directions
                    ]
                )
            ]
        elif self.mode == 1:
            # out["F"] = [self.T(X,'thresh',1) , self.T(X, 'window',1) , self.T(X, 'min_residuals',1)]
            out["F"] = [sum([self.T(X, d, 1) for d in self.generalisation_directions])]
            out["G"] = [self.falsification_count()]
        else:
            raise ValueError(f"Optimisation Mode {self.mode} not known.")


FitZScorePipeline = SaQCProblemJoin([LowpassResids, FitZScore])
FitGeneralPipeline = SaQCProblemJoin(
    [FitConstants, FitScatterLowpass, LowpassResids, FitZScore]
)


def make_problem(anomaly_spec: str | list[str]):
    """
    Get problem class from str (or merged problem class from list of strings).
    """
    # if anomaly spec is a list, assume it holds strings referring to problem classes,
    # that can be derived with "make_problems" to generate a ProblemSequence instance
    if isinstance(anomaly_spec, list | tuple):
        return SaQCProblemJoin([make_problem(p) for p in anomaly_spec])

    # so anomaly_spec is not referring to a sequence of problems:
    # if anomaly_spec is a string: try find a SaQCProblem
    # subclass named anomaly_spec in this module and return that
    glob = globals()
    problem = glob.get(anomaly_spec, None)
    if isinstance(problem, SaQCProblemJoin):
        return problem
    if (
        (problem is not None)
        and (type(problem) == abc.ABCMeta)
        and issubclass(problem, SaQCProblem)
    ):
        return problem

    # if no problem class named anomaly_spec exists, it may have been an anomaly alias
    problem = PROBLEM_ALIAS.get(anomaly_spec.lower(), None)
    if problem is not None:
        return glob[problem]

    # if spec also wasnt a registered anomaly type, i dont know what else it is meant to designate
    raise ValueError(
        f"Whats your Problem? It needs to be a Name of a SaQCProblem Subclass, a list of those, or one of {list(ANOMALY_ALIAS.keys())}. Got {anomaly_spec}."
    )


def merge_problems(problems):
    """
    Problem Class factory:
    Returns a MergedProblem class that is composed of the problem instances in Problems.
    """
    _merge_suffix = "__M"
    merge_enum = lambda x, ix, key=_merge_suffix: f"{x}{key}{ix}"

    def Xi_(X, i, key=_merge_suffix):
        i_key = f"{key}{i}"
        keys = [k for k in X if k.rfind(i_key) > 0]

        return {k[: k.rfind(i_key)]: X[k] for k in keys}

    class MergedProblem(SaQCProblem):
        def name(self):
            merge_seq = [p.name() for p in problems]
            return ">".join(merge_seq)

        def paraIni(self) -> dict:
            self.problems = problems
            self.parameters = {}
            self.isolation_directions = []
            for i, prob in enumerate(self.problems):
                self.parameters.update(
                    {
                        merge_enum(p, i): prob.parameters[p]
                        for p in prob.parameters.keys()
                    }
                )
                self.isolation_directions += [
                    merge_enum(d, i) for d in prob.isolation_directions
                ]
                if self.processing_type != "data":
                    self.processing_type = prob.processing_type

            def conversion(X):
                x = {}
                for i, prob in enumerate(problems):
                    conv_i = prob.conversions(Xi_(X, i))
                    x.update({merge_enum(key, i): conv_i[key] for key in conv_i})
                return x

            self.conversions = conversion
            bias = [None] * len(problems)
            for i, prob in enumerate(problems):
                i_bias = prob.getBias()
                bias[i] = [
                    {merge_enum(b, i): indiv[b] for b in indiv.keys()}
                    for indiv in i_bias
                ]
            merged_bias = [{}] * len(bias[0])
            for i, b in enumerate(zip(*tuple(bias))):
                for _b in b:
                    merged_bias[i] = dict(merged_bias[i], **_b)

            self.setBias(merged_bias, bias_portion=1.0)

            self.config = []
            confi_groups = []
            for i, prob in enumerate(problems):
                prob._tmp_suffix = "M"

                if len(prob.config) == 1:
                    prob.config[0][1].update(
                        {p: prob._configFunc(p) for p in prob.parameters}
                    )
                self.config += prob.config
                confi_groups += [i] * len(prob.config)

            if "data" in [p.processing_type for p in problems]:
                # processing happens on side lane:
                extra_f = self._configFunc("new_field")
                for i, c in zip(confi_groups, self.config):
                    for p, v in c[1].items():
                        if isinstance(v, Callable):
                            _fixFunc = lambda var, ex, X, v=v, extra_f=extra_f, i=i: v(
                                extra_f(var, ex, {}),
                                (extra_f(var, ex, {}),) + ex,
                                Xi_(X, i),
                            )
                            c[1][p] = _fixFunc
                    if "field" not in c[1]:
                        c[1].update({"field": extra_f})

                self.config = [("copyField", dict(target=extra_f))] + self.config
                self.config += [
                    (
                        "transferFlags",
                        dict(
                            field=extra_f,
                            target=self._configFunc("field"),
                            squeeze=False,
                        ),
                    )
                ]
                self.config += [("dropField", dict(field=extra_f))]

        def _processing_cache(self):
            return self.problems[0]._cache

        @staticmethod
        def funcBody(qc, field, X: dict, **kwargs) -> "SaQC":

            _qc = qc.copy()
            for i, prob in enumerate(problems):
                kwargs_i = dict(kwargs, label=kwargs.get("label", "") + f"_F{i}")
                if i > 0:
                    kwargs_i.pop("_processing_cache", None)
                _qc = prob.funcBody(_qc, field, Xi_(X, i), **kwargs_i)
            return _qc

        def objective(self, X: dict, out: dict):
            out_F = 0
            out_G = []
            _out = [{}] * len(self.problems)
            for i, prob in enumerate(self.problems):
                assert self.mode == prob.mode
                # update sub problems flags
                prob._flags = self._flags
                # update subproblems confusion
                prob._confusion()
                _out = {}
                prob.objective(Xi_(X, i), _out)

                out_F += _out["F"][0]
                if prob.mode == 0:
                    out_F -= prob.falsification_count()
                    out_F = out_F / len(self.problems)
                    out_G += _out.get("G", [])
                else:
                    # asserting falsification constraint is first!
                    out_G += _out.get("G", [])[1:]

            if self.mode == 0:
                out_F += self.falsification_count()
            else:
                out_G = [self.falsification_count()] + out_G

            out["F"] = [out_F]
            out["G"] = out_G

        def switchMode(self, res):
            for prob in self.problems:
                # prepare merged problems for being called by mode Switch (through self._evaluate):
                prob.mode = 1
                prob.n_ieq_constr += 1

            # switch mode of self
            super().switchMode(res)

            for prob in self.problems:
                # manually update merged problems targets:
                prob.target_flags = prob.TP
                prob.skip_mask = prob.FN | prob.FP | prob.skip_mask
            # make consistency check (expect faslification_count() be first constraint in all merged problems:

            self._evaluate(res.X, {})
            for i, prob in enumerate(self.problems):
                check_dict = {}
                prob.objective(Xi_(res.X, i), check_dict)
                assert check_dict["G"][0] == 0

    return MergedProblem
