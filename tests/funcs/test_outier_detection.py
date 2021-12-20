#! /usr/bin/env python
# -*- coding: utf-8 -*-

# see test/functs/fixtures.py for global fixtures "course_..."
import dios
from tests.fixtures import *

from saqc.funcs.outliers import (
    flagMAD,
    flagOffset,
    flagRaise,
    flagMVScores,
    flagByGrubbs,
    flagCrossStatistics,
)
from saqc.constants import *
from saqc.core import initFlagsLike


@pytest.fixture(scope="module")
def spiky_data():
    index = pd.date_range(start="2011-01-01", end="2011-01-05", freq="5min")
    s = pd.Series(np.linspace(1, 2, index.size), index=index, name="spiky_data")
    s.iloc[100] = 100
    s.iloc[1000] = -100
    flag_assertion = [100, 1000]
    return dios.DictOfSeries(s), flag_assertion


def test_flagMad(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result = flagMAD(data, field, flags, "1H", flag=BAD)
    flag_result = flags_result[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


def test_flagSpikesBasic(spiky_data):
    data = spiky_data[0]
    field, *_ = data.columns
    flags = initFlagsLike(data)
    data, flags_result = flagOffset(
        data, field, flags, thresh=60, tolerance=10, window="20min", flag=BAD
    )
    flag_result = flags_result[field]
    test_sum = (flag_result[spiky_data[1]] == BAD).sum()
    assert test_sum == len(spiky_data[1])


# see test/functs/fixtures.py for the 'course_N'
@pytest.mark.parametrize(
    "dat",
    [
        pytest.lazy_fixture("course_1"),
        pytest.lazy_fixture("course_2"),
        pytest.lazy_fixture("course_3"),
        pytest.lazy_fixture("course_4"),
    ],
)
def test_flagSpikesLimitRaise(dat):
    data, characteristics = dat()
    field, *_ = data.columns
    flags = initFlagsLike(data)
    _, flags_result = flagRaise(
        data,
        field,
        flags,
        thresh=2,
        freq="10min",
        raise_window="20min",
        numba_boost=False,
        flag=BAD,
    )
    assert np.all(flags_result[field][characteristics["raise"]] > UNFLAGGED)
    assert not np.any(flags_result[field][characteristics["return"]] > UNFLAGGED)
    assert not np.any(flags_result[field][characteristics["drop"]] > UNFLAGGED)


# see test/functs/fixtures.py for the 'course_N'
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_flagMVScores(dat):
    def _check(fields, flags, characteristics):
        for field in fields:
            isflagged = flags[field] > UNFLAGGED
            assert isflagged[characteristics["raise"]].all()
            assert not isflagged[characteristics["return"]].any()
            assert not isflagged[characteristics["drop"]].any()

    data1, characteristics = dat(
        periods=1000, initial_level=5, final_level=15, out_val=50
    )
    data2, characteristics = dat(
        periods=1000, initial_level=20, final_level=1, out_val=30
    )
    fields = ["field1", "field2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["field1", "field2"])
    flags = initFlagsLike(data)
    _, flags_result = flagMVScores(
        data=data,
        field=fields,
        flags=flags,
        trafo=np.log,
        iter_start=0.95,
        n=10,
        flag=BAD,
    )
    _check(fields, flags_result, characteristics)


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_3")])
def test_grubbs(dat):
    data, char_dict = dat(
        freq="10min",
        periods=45,
        initial_level=0,
        final_level=0,
        crowd_size=1,
        crowd_spacing=3,
        out_val=-10,
    )
    flags = initFlagsLike(data)
    data, result_flags = flagByGrubbs(
        data, "data", flags, window=20, min_periods=15, flag=BAD
    )
    assert np.all(result_flags["data"][char_dict["drop"]] > UNFLAGGED)


@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_flagCrossStatistics(dat):
    data1, characteristics = dat(initial_level=0, final_level=0, out_val=0)
    data2, characteristics = dat(initial_level=0, final_level=0, out_val=10)
    fields = ["field1", "field2"]
    s1, s2 = data1.squeeze(), data2.squeeze()
    s1 = pd.Series(data=s1.values, index=s1.index)
    s2 = pd.Series(data=s2.values, index=s1.index)
    data = dios.DictOfSeries([s1, s2], columns=["field1", "field2"])
    flags = initFlagsLike(data)

    _, flags_result = flagCrossStatistics(
        data, fields, flags, thresh=3, method=np.mean, flag=BAD
    )
    for field in fields:
        isflagged = flags_result[field] > UNFLAGGED
        assert isflagged[characteristics["raise"]].all()
