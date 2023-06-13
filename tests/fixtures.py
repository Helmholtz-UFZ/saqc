# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from saqc.core import DictOfSeries

# TODO: this is odd
#  Why not simple fixtures with talking-names,
#  that also take parameter, if needed


@pytest.fixture
def char_dict():
    return {
        "raise": pd.DatetimeIndex([]),
        "drop": pd.DatetimeIndex([]),
        "peak": pd.DatetimeIndex([]),
        "return": pd.DatetimeIndex([]),
        "missing": pd.DatetimeIndex([]),
    }


@pytest.fixture
def course_1(char_dict):
    """
    MONOTONOUSLY ASCENDING/DESCENDING

    values , that monotonously ascend towards a peak level, and thereafter do monotonously decrease
    the resulting drop/raise per value equals:  (peak_level - initial_level) / (0.5*(periods-2))
    periods number better be even!
    """

    def fix_funk(
        freq="10min",
        periods=10,
        initial_level=0,
        peak_level=10,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
        name="data",
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        left = np.linspace(initial_level, peak_level, int(np.floor(len(t_index) / 2)))
        right = np.linspace(peak_level, initial_level, int(np.ceil(len(t_index) / 2)))
        s = pd.Series(np.append(left, right), index=t_index)

        char_dict["raise"] = s.index[1 : int(np.floor(len(t_index) / 2))]
        char_dict["drop"] = s.index[int(np.floor(len(t_index) / 2) + 1) :]
        char_dict["peak"] = s.index[
            int(np.floor(len(t_index) / 2)) - 1 : int(np.floor(len(t_index) / 2)) + 1
        ]

        data = DictOfSeries({name: s})
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_2(char_dict):
    """
    SINGLE_SPIKE

    values , that linearly  develop over the whole timeseries, from "initial_level" to "final_level", exhibiting
    one "anomalous" or "outlierish" value of magnitude "out_val" at position "periods/2"
    number of periods better be even!
    """

    # SINGLE_SPIKE
    def fix_funk(
        freq="10min",
        periods=10,
        initial_level=0,
        final_level=2,
        out_val=5,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        data = np.linspace(initial_level, final_level, int(np.floor(len(t_index))))

        data = pd.Series(data=data, index=t_index)
        data.iloc[int(np.floor(periods / 2))] = out_val

        if out_val > data.iloc[int(np.floor(periods / 2) - 1)]:
            kind = "raise"
        else:
            kind = "drop"

        char_dict[kind] = data.index[int(np.floor(periods / 2))]
        char_dict["return"] = data.index[int(np.floor(len(t_index) / 2)) + 1]

        data = DictOfSeries(data=data)
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_test(char_dict):
    """
    Test function for pattern detection

    same as test pattern for first three values, than constant function
    """

    def fix_funk(
        freq="1 D",
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        out_val=5,
        char_dict=char_dict,
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=100)

        data = pd.Series(data=0, index=t_index)
        data.iloc[2] = out_val
        data.iloc[3] = out_val

        data = DictOfSeries(data=data)
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_3(char_dict):
    """
    CROWD IN A PIT/CROWD ON A SUMMIT

    values , that linearly  develop over the whole timeseries, from "initial_level" to "final_level", exhibiting
    a "crowd" of "anomalous" or "outlierish" values of magnitude "out_val".
    The "crowd/group" of anomalous values starts at position "periods/2" and continues with an additional amount
    of "crowd_size" values, that are each spaced "crowd_spacing" minutes from there predecessors.
    number of periods better be even!
    chrowd_size * crowd_spacing better be less then window[minutes].
    """

    def fix_funk(
        freq="10min",
        periods=10,
        initial_level=0,
        final_level=2,
        out_val=-5,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
        crowd_size=5,
        crowd_spacing=1,
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        data = np.linspace(initial_level, final_level, int(np.floor(len(t_index))))
        data = pd.Series(data=data, index=t_index)

        ind1 = data.index[int(np.floor(periods / 2))]
        dates = [
            ind1 + crowd_spacing * pd.Timedelta(f"{k}min")
            for k in range(1, crowd_size + 1)
        ]
        insertion_index = pd.DatetimeIndex(dates)

        data.iloc[int(np.floor(periods / 2))] = out_val
        data = pd.concat(
            [data, pd.Series(data=out_val, index=insertion_index)]
        ).sort_index()
        anomaly_index = insertion_index.insert(
            0, data.index[int(np.floor(periods / 2))]
        )

        if out_val > data.iloc[int(np.floor(periods / 2) - 1)]:
            kind = "raise"
        else:
            kind = "drop"

        char_dict[kind] = anomaly_index
        char_dict["return"] = t_index[int(len(t_index) / 2) + 1]

        data = DictOfSeries(data=data)
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_4(char_dict):
    """
    TEETH (ROW OF SPIKES) values

    , that remain on value level "base_level" and than begin exposing an outlierish or
    spikey value of magnitude "out_val" every second timestep, starting at periods/2, with the first spike. number
    of periods better be even!
    """

    def fix_funk(
        freq="10min",
        periods=10,
        base_level=0,
        out_val=5,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        data = pd.Series(data=base_level, index=t_index)
        data[int(len(t_index) / 2) :: 2] = out_val
        char_dict["raise"] = t_index[int(len(t_index) / 2) :: 2]
        char_dict["return"] = t_index[int((len(t_index) / 2) + 1) :: 2]

        data = DictOfSeries(data=data)
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_5(char_dict):
    """
    NAN_holes

    values , that ascend from initial_level to final_level linearly and have missing data(=nan)
    at positions "nan_slice", (=a slice or a list, for iloc indexing)
    periods better be even!
    periods better be greater 5
    """

    def fix_funk(
        freq="10min",
        periods=10,
        nan_slice=slice(0, None, 5),
        initial_level=0,
        final_level=10,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
    ):
        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        values = np.linspace(initial_level, final_level, periods)
        s = pd.Series(values, index=t_index)
        s.iloc[nan_slice] = np.nan
        char_dict["missing"] = s.iloc[nan_slice].index

        data = DictOfSeries(data=s)
        return data, char_dict

    return fix_funk
