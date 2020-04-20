import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def char_dict():
    return {
        "raise": pd.DatetimeIndex([]),
        "drop": pd.DatetimeIndex([]),
        "peak": pd.DatetimeIndex([]),
        "return": pd.DatetimeIndex([]),
    }


@pytest.fixture
def course_1(char_dict):
    # MONOTONOUSLY ASCENDING/DESCENDING
    # values , that monotonously ascend towards a peak level, and thereafter do monotonously decrease
    # the resulting drop/raise per value equals:  (peak_level - initial_level) / (0.5*(periods-2))
    # periods number better be even!
    def fix_funk(
        freq="10min",
        periods=10,
        initial_level=0,
        peak_level=10,
        initial_index=pd.Timestamp(2000, 1, 1, 0, 0, 0),
        char_dict=char_dict,
    ):

        t_index = pd.date_range(initial_index, freq=freq, periods=periods)
        data = np.append(
            np.linspace(initial_level, peak_level, int(np.floor(len(t_index) / 2))),
            np.linspace(peak_level, initial_level, int(np.ceil(len(t_index) / 2))),
        )
        data = pd.DataFrame(data=data, index=t_index, columns=["data"])
        char_dict["raise"] = data.index[1 : int(np.floor(len(t_index) / 2))]
        char_dict["drop"] = data.index[int(np.floor(len(t_index) / 2) + 1) :]
        char_dict["peak"] = data.index[int(np.floor(len(t_index) / 2)) - 1 : int(np.floor(len(t_index) / 2)) + 1]
        return data, char_dict

    return fix_funk


@pytest.fixture
def course_2(char_dict):
    # SINGLE_SPIKE
    # values , that linearly  develop over the whole timeseries, from "initial_level" to "final_level", exhibiting
    # one "anomalous" or "outlierish" value of magnitude "out_val" at position "periods/2"
    # number of periods better be even!
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
        return data.to_frame("data"), char_dict

    return fix_funk


@pytest.fixture
def course_3(char_dict):
    # CROWD IN A PIT/CROWD ON A SUMMIT
    # values , that linearly  develop over the whole timeseries, from "initial_level" to "final_level", exhibiting
    # a "crowd" of "anomalous" or "outlierish" values of magnitude "out_val".
    # The "crowd/group" of anomalous values starts at position "periods/2" and continues with an additional amount
    # of "crowd_size" values, that are each spaced "crowd_spacing" minutes from there predecessors.
    # number of periods better be even!
    # chrowd_size * crowd_spacing better be less then freq[minutes].
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
        insertion_index = pd.DatetimeIndex(
            [ind1 + crowd_spacing * pd.Timedelta(str(k) + "min") for k in range(1, crowd_size + 1)]
        )
        data.iloc[int(np.floor(periods / 2))] = out_val
        data = data.append(pd.Series(data=out_val, index=insertion_index))
        data.sort_index(inplace=True)
        anomaly_index = insertion_index.insert(0, data.index[int(np.floor(periods / 2))])
        if out_val > data.iloc[int(np.floor(periods / 2) - 1)]:
            kind = "raise"
        else:
            kind = "drop"
        char_dict[kind] = anomaly_index
        char_dict["return"] = t_index[int(len(t_index) / 2) + 1]
        return data.to_frame("data"), char_dict

    return fix_funk


@pytest.fixture
def course_4(char_dict):
    # TEETH (ROW OF SPIKES)
    # values , that remain on value level "base_level" and than begin exposing an outlierish or spikey value of magnitude
    # "out_val" every second timestep, starting at periods/2, with the first spike.
    # number of periods better be even!
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
        return data.to_frame("data"), char_dict

    return fix_funk
