#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# see test/functs/fixtures.py for global fixtures "course_..."
import pytest

import saqc
from saqc import BAD, UNFLAGGED, DictOfSeries, SaQC
from saqc.core.flags import initFlagsLike
from saqc.lib.tools import getHist
from tests.fixtures import char_dict, course_1, course_2, course_3, course_4

SEED = 42
DATLEN = 1000
DATA = pd.Series(
    np.sin(0.1 * np.arange(DATLEN)),
    index=pd.date_range("2000", freq="10min", periods=DATLEN),
    name="data",
)


def addOutliers(
    dat, scope=None, outliers_num=10, outliers_range=10, outliers_min=20, seed=SEED
):
    np.random.seed(seed)
    scope = scope or (0, len(dat))
    outliers_ilocs = np.random.choice(np.arange(*scope), outliers_num, False)
    outliers_offsets = np.array(
        [
            np.random.choice([-1, 1])
            * (outliers_min + np.random.random() * outliers_range)
            for k in range(outliers_num)
        ]
    )
    for i in enumerate(outliers_ilocs):
        dat.iloc[i[1]] = dat.iloc[i[1]] + outliers_offsets[i[0]]
    flags = pd.Series(False, index=dat.index)
    flags.iloc[outliers_ilocs] = True
    return dat, flags


def addNoise(
    dat, scope=None, noise_num=2, noise_amp=10, noise_len=(500, 600), seed=SEED
):
    np.random.seed(seed)
    scope = scope or (0, len(dat))
    noise_ilocs = np.random.choice(
        np.arange(scope[-1] - (2 * noise_len[-1])), noise_num, False
    )
    flags = pd.Series(False, index=dat.index)
    for i in enumerate(noise_ilocs):
        noise_idx = np.arange(i[1], i[1] + np.random.choice(np.arange(*noise_len)))
        noise = (
            np.zeros(len(noise_idx))
            + (np.random.random(len(noise_idx)) - 0.5) * noise_amp
        )
        dat.iloc[noise_idx] = dat.iloc[noise_idx] + noise
        flags.iloc[noise_idx] = True
    return dat, flags


def addConstants(
    dat, scope=None, const_num=2, const_len=(500, 600), const_lvl=None, seed=SEED
):
    np.random.seed(seed)
    scope = scope or (0, len(dat))
    const_ilocs = np.random.choice(
        np.arange(scope[-1] - (2 * const_len[-1])), const_num, False
    )
    flags = pd.Series(False, index=dat.index)
    for i in enumerate(const_ilocs):
        const_idx = np.arange(i[1], i[1] + np.random.choice(np.arange(*const_len)))
        const = np.zeros(len(const_idx)) + (const_lvl or dat.iloc[i[1]])
        dat.iloc[const_idx] = const
        flags.iloc[const_idx] = True
    return dat, flags


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_single_target_outlier():
    data, outliers = addOutliers(DATA)
    qc = saqc.SaQC(data)
    qc = qc.copyField("data", "data_unflagged")
    qc = qc.setFlags("data", data=outliers[outliers].index, label="outlier").supervise(
        "data", problem_labels=["outlier"]
    )
    qc = qc.calibratePipeline(
        "data",
        problems=["outlierz"],  # alias for ZScore
        name="zOD",
        pop_size=5,
        termination=("n_evals", 10),
    )
    qc = qc.calibratePipeline(
        "data",
        problems=["outlier"],  # alias for uniLOF
        name="lofOD",
        pop_size=5,
        termination=("n_evals", 10),
    )
    qc = qc.calibratePipeline(
        "data",
        problems=["range"],
        name="rangOD",
        pop_size=5,
        termination=("n_evals", 10),
    )
    flagged = qc.zOD("data_unflagged").flags["data_unflagged"] > 0
    assert np.all(outliers.values == flagged.values)
    flagged = qc.lofOD("data_unflagged").flags["data_unflagged"] > 0
    assert np.all(outliers.values == flagged.values)
    flagged = qc.rangOD("data_unflagged").flags["data_unflagged"] > 0
    assert np.all(outliers.values == flagged.values)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_single_target_noise():
    data, flags = addNoise(DATA, noise_len=(30, 50), noise_num=3, noise_amp=1)
    qc = saqc.SaQC(data)
    qc = qc.copyField("data", "data_unflagged")
    qc = qc.setFlags("data", data=flags[flags].index, label="noise").supervise(
        "data", problem_labels=["noise"]
    )
    qc = qc.calibratePipeline(
        "data",
        problems=[["filter", "noise"]],
        name="flagNoise",
        pop_size=10,
        termination=("n_evals", 100),
    )
    flagged = qc.flagNoise("data_unflagged").flags["data_unflagged"] > 0
    # assert np.sum(flags.values == flagged.values) > DATLEN - 20


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_single_target_constants():
    data, flags = addConstants(DATA, const_num=2, const_len=(10, 20))
    qc = saqc.SaQC(data)
    qc = qc.copyField("data", "data_unflagged")
    qc = qc.setFlags("data", data=flags[flags].index, label="constant").supervise(
        "data", problem_labels=["constant"]
    )
    qc = qc.calibratePipeline(
        "data",
        problems=["constant"],
        name="flagConstant",
        pop_size=10,
        termination=("n_evals", 100),
    )
    flagged = qc.flagConstant("data_unflagged").flags["data_unflagged"] > 0
    # assert np.all(flags.values == flagged.values)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_multi_target():
    data, flags_const = addConstants(
        DATA, const_num=2, const_len=(10, 20), seed=SEED + 1
    )
    data, flags_noise = addNoise(
        data, noise_len=(30, 50), noise_num=3, noise_amp=3, seed=SEED + 2
    )
    data, flags_outliers = addOutliers(data, seed=SEED + 3)
    qc = saqc.SaQC(data)
    qc = qc.copyField("data", "data_unflagged")
    qc = qc.setFlags(
        "data",
        data=flags_const[flags_const].index,
        label="constant",
        override=True,
    )
    qc = qc.setFlags(
        "data",
        data=flags_noise[flags_noise].index,
        label="noise",
        override=True,
    )
    qc = qc.setFlags(
        "data",
        data=flags_outliers[flags_outliers].index,
        label="outlier",
        override=True,
    )

    qc = qc.calibratePipeline(
        "data",
        problems=["constant", "noise", "outlier"],
        problem_labels=["constant", "noise", "outlier"],
        name="flagPipeline1",
        pop_size=10,
        termination=("n_evals", 100),
    )
    qc = qc.calibratePipeline(
        "data",
        problems=[["constant", "noise", "outlierz"]],
        problem_labels=[["constant", "noise", "outlier"]],
        name="flagPipeline2",
        pop_size=20,
        termination=("n_evals", 1000),
    )
    flagged = qc.flagPipeline1("data_unflagged").flags["data_unflagged"] > 0
    matches = (qc["data"].flags["data"] > 0) == flagged
    # assert matches.sum() > DATLEN - 50
    flagged = qc.flagPipeline2("data_unflagged").flags["data_unflagged"] > 0
    matches = (qc["data"].flags["data"] > 0) == flagged
    # assert matches.sum() > DATLEN - 50
