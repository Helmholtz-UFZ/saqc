#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymoo.core.result import Result
from pymoo.optimize import minimize

from saqc.core import flagging, processing, register
from saqc.lib.problems import make_problem
from saqc.lib.tools import getHist, getHistByKwarg, getMeta

if TYPE_CHECKING:
    from saqc import SaQC
# control the SEED for initial population sampling
OPT_SEED = None


class OptiSaqcMixin:
    @register(mask=[], demask=[], squeeze=[], handles_target=True)
    def supervise(
        self,
        field: str,
        problem_labels: list[str | None],
        override: bool = True,
        target: str = None,  # target funcs as supervision source (take problem_labeled flags from target and assign to field)
        **kwargs,
    ):  # TODO only mask/penalize scores at flags that are no targets for subsequent problems
        """
        Supervise data, so saqc parameter estimation can be ran against it.

        The function annotates specific flags (columns) as being the Ground Truth (True Positives)
        For subsequent calibration of flagging function pipeline.

        Supervise drops all history columns/flags that are not listed as problem labels.

        Parameters
        ----------
        problem_labels :
            A list of anomaly types, data gets to be classified by.

            Any label that does not appear in field history, will cause pop up GUI where targets for this anomaly type
            can be assigned.
            Pass ``None`` label triggers semi supervised fit (Fitting without flags).
            Join labels to a single target by summarizing them as list items.

        override :
            If True (default), target is overridden with variable and its supervised history, (if target already exists.)
            If False, target gets appended supervised history, (if target already exists).
        """
        # initial count for supervision chain
        supervision_state = 0
        # unpack labels and make merge group mapping
        flat_labels = []
        merge_groups = []
        for i, p in enumerate(problem_labels):
            if isinstance(p, str) or (p is None):
                flat_labels += [p]
                merge_groups += [i]
            else:  # p is a list/tuple of string labels
                flat_labels += p
                merge_groups += [i] * len(p)

        if (len(flat_labels) > len(problem_labels)) and (not override):
            raise NotImplementedError(
                "Appending to partly supervised field not implemented with merged labels."
            )

        # collecting all flags that are already assigned to field under a problem label name
        trg_flags = []
        for l in flat_labels:
            if l.startswith("__hloc"):
                trg_flags.append(getHist(self[field], int(l[6:])))
            else:
                trg_flags.append(getHistByKwarg(self[field], l)[0])
        # trg_flags = [getHistByKwarg(self[field], l)[0] for l in flat_labels]
        target = target or field

        if target == field:
            if not override:
                raise NotImplementedError(
                    "supervision without override is not implemented yet"
                )
            else:
                # if target=field shall get overridden, re init fresh field
                self[field] = self.__class__(self.data[field].rename(field))
        elif target in self.columns:  # if target is an already existing field
            if not override:  # if append is performed
                raise NotImplementedError(
                    "supervision without override is not implemented yet"
                )
            else:  # re init target
                self[target] = type(self)(self.data[field].rename(target))
        elif target not in self.columns:  # new target field
            self[target] = self.__class__(self.data[field].rename(target))

        group = 0
        member = 0
        # now iterate over flags list and labels, to generate supervised history entries/spawn GUI where no flags are assigned
        for i, (f, l, g) in enumerate(zip(trg_flags, flat_labels, merge_groups)):
            sv = supervision_state
            if (
                g == group
            ):  # merge group current label belongs to, is the same as last iteration
                _sv = sv + float(f"{g}.{member}")
                member += 1
            else:
                group += 1
                _sv = sv + float(f"{g}.{0}")
                member = 1

            i_kwargs = dict(label=l or f"None_{_sv}", supervised=_sv, **kwargs)

            if (f is None) and (l is not None):  # need user input via GUI
                i_kwargs.pop("dfilter")
                i_kwargs.update(
                    dict(ax_kwargs=dict(title=f'Assign "{l}" type anomalies'))
                )
                # GUI GUI GUI
                self = self.flagByClick(
                    target,
                    gui_mode="overlay",
                    **i_kwargs,
                )
                # if GUI was clased without assigning flags
                if getHistByKwarg(self, i_kwargs["label"])[0] is None:
                    # make UNFLAGGED dummy column for consistency
                    self = self.flagDummy(target, **i_kwargs)

            else:  # this target label already has flags assigned, add them to the history (through set flags call)
                f = f if not (f is None) else pd.Series(False, index=self.data[target])
                is_trg_flagged = f.squeeze() > kwargs["dfilter"]
                if (
                    is_trg_flagged.any()
                ):  # need to check that because empty index objects make setFlags whine
                    self = self.setFlags(
                        target,
                        data=is_trg_flagged[is_trg_flagged].index,
                        **i_kwargs,
                    )
                else:
                    self = self.flagDummy(target, **i_kwargs)
        return self

    @register(mask=[], demask=[], squeeze=[])
    def calibratePipeline(
        self: "SaQC",
        field: str,
        problems: list[str],
        name: str = None,
        problem_labels: list[str] = None,
        pop_size: int = 100,
        termination: tuple | int = None,
        log_pop: bool = None,
        log_config: bool = None,
        log_path: str = None,
        verbose: bool = True,
        **kwargs,
    ) -> "SaQC":
        """
        Optimize problem pipeline against supervised field.

        Parameters
        ----------
        problems :
            Problem Statement of the Pipeline.
            Definition of the pipeline in terms of a list of (possibly merged) Problems.

        name :
            Name of the Pipeline
            Sets the name, the resulting Pipeline is accessible through, as a new SaQC method.

        problem_labels :
            Target (column) Labels for the Pipeline.
            If None (default): all flags get squashed to a merged target (possibly to be iterated over by sequential pipeline)
            If []: fallback to supervision.
            If a list of length thats matching the length of the problems list, problems will be optimised against it sequentially
            If length of labels list is exactly 1 (or None), all problems will be fit against this single label sequencially.
            If a label is listed, thats not present in the history, GUI assignment is initialised.

        log_pop :
            Weather to log the whole population generated during training to ``log_path``.
            Defaults to `True` as long as valid ``log_path`` is given.

        log_config :
            Weather to log the optimized parameterset as a config file to ``log_path``.
            Defaults to `True` as long as valid ``log_path`` is given.

        log_path :
            Path to the Logs. (folder path), If not given (None) logging wont happen.
            If path already exists, content will be overridden.

        pop_size :
            Population Size to optimize with.

        termination :
            Determine Termination for the Optimisations.
            If ``None`` (default) - fall back to pymoo defaults (computationally exhaustive, but most likely not
            terminating too early).
            If an integer, will be interpreted simply as the maximal number of evaluations.

        verbose :
            Logging verbosity.
            Wanna see whats happening while waiting for Results?

        Notes
        -----
        Either ``name`` or both, ``log_path`` and ``config_path`` should be assigned since otherwise optimised pipeline gets
        lost after run.

        """
        if pop_size is None:
            pop_size = 100 * len(problems)
        if termination is None:
            termination = 20 * pop_size
        if isinstance(termination, int):
            termination = ("n_evals", termination)

        if problem_labels is None:
            problem_labels = []
            metas = getMeta(self)
            for m in enumerate(metas):
                _p_label = m[1].get("kwargs", {}).get("label", None)
                if _p_label is None:
                    _p_label = f"__hloc{m[0]}"
                problem_labels.append(_p_label)
            problem_labels = [problem_labels]

        if (len(problem_labels) == 1) and (len(problems) > 1):
            problem_labels = [problem_labels[0] for k in range(len(problems))]
        _pop_size = {"pop_size": pop_size}
        _termination = {}
        _dat_slice = slice(kwargs.get("start_date", None), kwargs.get("end_date", None))
        if termination is not None:
            _termination = {"termination": termination}

        if log_path is None:
            log_pop = False
            log_config = False
        else:
            if not os.path.exists(log_path):
                os.makedirs(log_path, exist_ok=True)

            log_config = True if log_config is None else log_config
            log_pop = True if log_pop is None else log_pop

        if (name is None) and (not log_config):
            raise ValueError(
                f'Parameter "name" was not assigned and "log_config" evaluates to False. '
                f"Optimisation Results wont be accessible."
            )
        p_n = len(problems)

        if (
            len(problem_labels) == 0
        ):  # if no problem targets were passed, assume field is already supervised
            sv_field = field
            problem_labels = [None] * p_n
        else:  # generate new temporal field to hold supervised data
            sv_field = "VAR" + str(uuid.uuid4()).replace("-", "_")
            self = self.supervise(field, problem_labels, target=sv_field)

        opt_results = [None] * p_n
        falsific_score = [None] * p_n
        problem_instances = [None] * p_n
        problem_classes = [None] * p_n
        trg_flags = [None] * p_n
        trg_join = pd.Series(False, index=self.data[sv_field][_dat_slice].index)

        # can now assume field_sv is supervised
        for i_sv in range(p_n):
            # get index of history col supervised at stage i_sv
            sub_group = 0
            col = []
            i_col = []
            while sub_group >= 0:
                _col, _i_col = getHistByKwarg(
                    self[sv_field], float(f"{i_sv}.{sub_group}"), "supervised"
                )
                if _col is None:
                    sub_group = -1
                else:
                    sub_group += 1
                    _col = _col[_dat_slice]
                    col += [_col]
                    i_col += [_i_col]

            trg_flags[i_sv] = pd.Series(
                False, index=self.data[sv_field][_dat_slice].index
            )
            label = []
            for c, i_c in zip(col, i_col):
                trg_flags[i_sv] |= c > kwargs["dfilter"]

                label += [getMeta(self[sv_field], i_c)["kwargs"]["label"]]

            problem_labels[i_sv] = " & ".join(label) if (len(label) > 1) else label[0]
            trg_join |= trg_flags[i_sv]

        # prepare a SaQC object to solve the problems on
        qc_opt = type(self)()
        qc_opt[sv_field] = type(self)(self.data[sv_field][_dat_slice].rename(sv_field))

        # solve problems one by one:
        for i, p in enumerate(problems):  # iterate over problems
            problem_classes[i] = make_problem(p)
            problem_instances[i] = problem_classes[i](
                qc=qc_opt,
                target_flags=trg_flags[i],
                skip_mask=trg_join & ~trg_flags[i],
                algorithm_kwargs=_pop_size,
                optimizer_kwargs=_termination,
            )

            if verbose:
                _printProblemOverview(
                    problem_instances[i], problem_labels[i], trg_flags[i]
                )

            while True:
                # problem helps with solving it:
                optimizer_paras = problem_instances[i].optimizerIni()

                opt_results[i] = minimize(
                    problem=problem_instances[i],
                    seed=OPT_SEED,
                    save_history=log_pop,
                    verbose=verbose,
                    **optimizer_paras,
                )

                if opt_results[i] is None:
                    return self

                if not hasattr(opt_results[i], "X"):
                    raise NotImplementedError("No valid solution found.")

                if falsific_score[i] is None:
                    falsific_score[i] = int(opt_results[i].F[0])

                if verbose:
                    _printProblemSolution(
                        problem_instances[i],
                        problem_labels[i],
                        falsific_score[i],
                        opt_results[i],
                    )

                if problem_instances[i].mode == 0:
                    problem_instances[i].switchMode(opt_results[i])
                else:
                    # try to free cache...
                    if hasattr(problem_instances[i]._cache, "cache_clear"):
                        problem_instances[i]._cache.cache_clear()
                        problem_instances[i]._cache = None
                    break

            # TODO: Here would be the place to NaNFill/Impute fitted problem targets.

        if verbose:
            _printSequenceSolution(
                problem_instances, problem_labels, falsific_score, opt_results
            )

        if log_pop:  # write optimisation final state and population to file
            _problem_log(opt_results, log_path)

        if log_config:  # write config with optimised parameters to file
            _config_log(
                log_path, problem_instances, problem_labels, opt_results, verbose
            )

        if name is not None:  # add the optimised pipeline as a method to ``SaQC`` class

            def pipeFunc(self: "SaQC", field: str, **kwargs):
                # use the label passed with the call (if one was passed) as prefix for the pipeline labels:
                _label = kwargs.pop("label", "")
                _label = _label + "_" if len(_label) > 0 else _label
                # problem by problem:
                for i, p in enumerate(problem_instances):
                    # retrieve conversion from opt space to saqc space
                    conv = problem_instances[i].conversions
                    # adjust label assigned with the problem
                    anomaly_type = problem_labels[i]
                    _kwargs = dict(label=_label + anomaly_type, **kwargs)
                    # if problem is flags assigning only, just append problems func body to the callchain
                    if p.processing_type == "flags":
                        self = p.funcBody(
                            self, field, conv(opt_results[i].X), **_kwargs
                        )
                    # if problem manipulates data values, assign the problem a side chain for processing and transfer
                    # flags onto targeted field afterwards
                    elif p.processing_type == "data":
                        tmp_field = "VAR" + str(uuid).replace("-", "_")
                        self = self.copyField(field, tmp_field)
                        self = p.funcBody(
                            self, tmp_field, conv(opt_results[i].X), **_kwargs
                        )
                        self = self.transferFlags(
                            tmp_field, field, squeeze=False
                        ).dropField(tmp_field)
                    else:
                        raise ValueError(f"Whats your problem? Got {p.processing_type}")

                return self

            setattr(pipeFunc, "__name__", name)
            processing()(pipeFunc)
        if field != sv_field:
            self = self.dropField(sv_field)
        return self

    @flagging()
    def applyConfig(self, field: str, path: str, name: str = None, **kwargs):
        """
        Apply the processing/flagging pipeline represented by an univariate config File to field, or "instantiate" the
        config file as a saqc method.

        Univariat Config File:
        * depends on only one input field for generation of all intermediary results/processings
        * flagging/processing result is all assigned/represented by final flagging/data status of the input field
        * all configs generated from SaQCProblem chains are univariat configs

        Parameters
        ----------
        path :
            Path to the config file to load.

        name :
            If given, the process representing the config file will be added to the ``saqc`` methods and be accesible via ``"name"``.
            In this case execution of the Algorithm onto field wont be performed.
        """

        # late import to avoid circular shenannigans
        from saqc.core import Flags
        from saqc.parsing.reader import _ConfigReader, readFile

        # get pd dataframe representation of the config
        df_repr = readFile(path)

        # determine the first field in varnames to be the focused input/output field (always correct for SaQCProblem generated configs)
        _field = df_repr["varname"].iloc[0]

        # load config as list of rows
        config = []
        with open(path, "r") as file:
            for line in file:
                config += [line.strip()]

        # define a function that executes the config and assigns the results to field
        def configFunc(qc, field, config_str="\n".join(config), **kwargs):

            d = qc.data[field].rename(_field)
            f = Flags(qc.flags[field].rename(_field).to_frame())
            tmp_field = "VAR" + str(uuid.uuid4()).replace("-", "_")
            qc[tmp_field] = _ConfigReader(d, f).readString(config_str).run()[_field]
            qc = qc.transferFlags(tmp_field, field, squeeze=False)
            qc = qc.dropField(tmp_field)
            return qc

        # add function as method to saqc class?
        if name is not None:
            setattr(configFunc, "__name__", name)
            processing()(configFunc)
            return self
        # or just execute it and obtain results for field
        else:
            return configFunc(self, field)


def _fitness_to_frame(M: np.ndarray | float, metric: str):
    """Helps writing populations"""
    M = np.array(M) if not isinstance(M, np.ndarray) else M
    M = M.reshape(M.shape[0], 1) if (len(M.shape) < 2) else M
    M_cols = [f"{metric}{k}" for k in range(M.shape[1])]
    return pd.DataFrame(M, columns=M_cols)


def _cvars_to_frame(X: np.ndarray):
    if X is None:
        return None
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    X = X.reshape(1) if (len(X.shape) < 1) else X

    X_cols = [k for k in X[0].keys()]
    X = np.array([list(x.values()) for x in X])
    return pd.DataFrame(X, columns=X_cols)


def _population_write(opt: Result, path: str):
    non_dominated_log = _cvars_to_frame(opt.X)
    if non_dominated_log is None:
        print("PARETO FRONT EMPTY")
        return 0
    else:
        non_dominated_log = [non_dominated_log]
    non_dominated_log += [
        _fitness_to_frame(getattr(opt, metric), metric) for metric in ["F", "G", "CV"]
    ]
    non_dominated_frame = pd.concat(non_dominated_log, axis=1)
    non_dominated_frame.to_csv(os.path.join(path, "non_dominated.csv"))
    if hasattr(opt, "pop"):  # optimisation history was recorded
        population_log = [_cvars_to_frame(opt.pop.get("X"))]
        for metric in ["F", "G", "CV"]:
            try:
                population_log += [_fitness_to_frame(opt.pop.get(metric), metric)]
            except ValueError:
                continue
        pd.concat(population_log, axis=1).to_csv(os.path.join(path, "population.csv"))

    return 0


def _problem_log(opt_results, log_opt):
    for i, opt in enumerate(opt_results):
        _path = os.path.join(log_opt, f"problem_{i}")
        if not os.path.exists(_path):
            os.mkdir(_path)
        _population_write(opt, _path)


def _individualStr(X):
    return "\n".join([f"    {p[0]}: {p[1]}" for p in X.items()])


def _sectionString(s):
    s = s[:-1] if s[-1:] == "\n" else s
    s = s[1:] if s[0] == " " else s
    line = len(s) * "-"
    return f" {line}\n {s}\n {line}\n"


def _printProblemSolution(problem_instance, problem_label, falsific_score, opt_result):
    head_str = (
        f" {problem_instance.name()}(mode={problem_instance.mode}) -> {problem_label}\n"
    )
    opt_str = f" FP + FN: {falsific_score}:\n"
    par_str = f" Parameters:\n"
    print(
        _sectionString(head_str)
        + opt_str
        + par_str
        + _individualStr(opt_result.X)
        + "\n"
    )


def _printProblemOverview(problem_instance, problem_label, trg_flags):
    head_str = f" {problem_instance.name()} -> {problem_label}\n"
    head_str = _sectionString(head_str)
    flag_str = f" Target: {problem_label} ({trg_flags.sum()} of {len(trg_flags)})\n"
    problem_string = problem_instance.problemInfo()
    optInfo = head_str + "\n" + flag_str + problem_string + "\n"
    print(optInfo)


def _printSequenceSolution(
    problem_instances, problem_labels, falsific_score, opt_results
):
    print("\n")
    print(_sectionString("Joint Optimum:"))
    for i in range(len(opt_results)):
        problem_str = f" {problem_instances[i].name()} -> {problem_labels[i]} (False = {falsific_score[i]}):\n"
        print(problem_str + _individualStr(opt_results[i].X) + "\n")


def _config_log(log_path, problem_instances, problem_labels, opt_results, verbose):
    """
    Generates the config file (string) from the problems fixated config attributes
    by
        * assigning fields/targets to varname column/signature positions
        * adding keywords that were implicitly set with their default values to the signatures.
    """
    config_path = os.path.join(log_path, "config.csv")
    var = problem_instances[0].var
    ex = tuple()
    varnames = []
    funcs = []
    fixed_config = []
    for i, prob in enumerate(problem_instances):
        C = prob.getFixedConfig(var, ex, opt_results[i].X, problem_labels[i])
        for c in C:  # for c in C (=list of config line represetations)
            funcs += [c[0]]  # collect function name
            nC = c[1].copy()  # retrieve function signature dictionary
            if "field" not in c[1]:  # if signature does not contain field:
                varnames += [
                    var
                ]  # add vield to the list of entries for the 'varname' column
            else:  # if field in signature
                if (
                    "target" in c[1]
                ):  # if target also listed in signature, who is going to varname column?
                    if isinstance(c[1]["field"], list):  # if field is a list,
                        varnames += [nC.pop("target")]  # target goes to varname column
                    else:  # if field is not a list
                        varnames += [nC.pop("field")]  # field goes to varname column
                else:
                    varnames += [
                        nC.pop("field")
                    ]  # if target is not listed in signature, field goes to varname column

            fixed_config += [nC]  # add modified signature dictionary
        ex += (
            prob.used_names
        )  # add the names used in the dictionary config to the list of used names

    # compose the signature string from varnames, functions and signatures list
    var_indent = max([len(v) for v in varnames]) + 1
    entries = ["varname" + " " * (var_indent - 7) + ";" + "test"]
    entries += ["#" + "-" * (var_indent - 1) + ";" + "-" * 50]
    for v, f, p in zip(varnames, funcs, fixed_config):
        v_str = v + " " * (var_indent - len(v))
        sig_str = []
        for k, v in p.items():
            _v = v
            if isinstance(_v, str):
                if not ((f in ["processGeneric", "flagGeneric"]) and (k == "func")):
                    _v = f"'{_v}'"

            sig_str += [f"{k}={_v}"]
        sig_str = ", ".join(sig_str)
        func_str = f"{f}({sig_str})"
        entries += [v_str + ";" + func_str]
    if verbose:
        print("\n".join(entries))
    with open(config_path, "w") as file:
        for item in entries:
            file.write(f"{item}\n")
