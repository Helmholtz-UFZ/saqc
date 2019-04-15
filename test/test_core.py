#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from core import runner, flagNext, prepareMeta
from config import Fields
from flagger import SimpleFlagger, DmpFlagger, PositionalFlagger
from test.common import initData


TESTFLAGGERS = [
    SimpleFlagger(), DmpFlagger(),  # PositionalFlagger()
]


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_temporalPartitioning(flagger):
    """
    Check if the time span in meta is respected
    """
    data = initData(3)
    var1, var2, var3, *_ = data.columns
    split_date = data.index[len(data.index)//2]
    tests = ["range, {min: -2, max: -1}",
             "generic, {func: this <= sum(this)}",
             "generic, {func: this <= sum(this)}"]

    meta = prepareMeta(
        pd.DataFrame(
            {Fields.VARNAME: [var1, var2, var3],
             Fields.STARTDATE: [None, None, split_date],
             Fields.ENDDATE: [None, split_date, None],
             Fields.FLAGS: tests}),
        data)

    pdata, pflags = runner(meta, flagger, data)

    fields = [Fields.VARNAME, Fields.STARTDATE, Fields.ENDDATE]
    for _, row in meta.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflags.loc[flagger.isFlagged(pflags[vname]), vname]
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_missingConfig(flagger):
    """
    Test if variables available in the dataset but not the config
    are handled correctly, i.e. are ignored
    """
    data = initData(2)
    var1, var2, *_ = data.columns
    meta = prepareMeta(
        pd.DataFrame(
            {Fields.VARNAME: [var1],
             Fields.FLAGS: ["range, {min: -9999, max: 9999}"]}),
        data)

    pdata, pflags = runner(meta, flagger, data)
    assert var1 in pdata and var2 not in pflags


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_missingVariable(flagger):
    """
    Test if variables available in the config but not dataset
    are handled correctly, i.e. are ignored
    """
    data = initData(1)
    var, *_ = data.columns
    meta = prepareMeta(
        pd.DataFrame(
            {Fields.VARNAME: [var, "empty"],
             Fields.FLAGS: ["range, {min: -9999, max: 9999}",
                            "range, {min: -9999, max: 9999}"]}),
        data)

    pdata, pflags = runner(meta, flagger, data)
    assert (data.columns == [var]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_assignVariable(flagger):
    """
    Test the assign keyword, a variable present in the configuration, but not
    dataset will be added to output flags
    """
    data = initData(1)
    var1, *_ = data.columns
    var2 = "empty"
    meta = prepareMeta(
        pd.DataFrame(
            {Fields.VARNAME: [var1, var2],
             Fields.FLAGS: ["range, {min: -9999, max: 9999}",
                            f"generic, {{func: isflagged({var2}), assign: True}}"]}),
        data)

    pdata, pflags = runner(meta, flagger, data)

    if isinstance(pflags.columns, pd.MultiIndex):
        cols = (pflags
                .columns.get_level_values(0)
                .drop_duplicates())
        assert (cols == [var1, var2]).all()
    else:
        assert (pflags.columns == [var1, var2]).all()


@pytest.mark.parametrize("flagger", TESTFLAGGERS)
def test_flagNext(flagger):
    """
    Test if the flagNext functionality works as expected
    """
    data = initData().iloc[:, 1]
    flags = flagger.initFlags(data)

    idx = [0, 1, 2]
    flags.iloc[idx] = flagger.setFlag(flags.iloc[idx])

    n = 4
    fflags = flagNext(flagger, flags.copy(), 4)
    result_idx = np.unique(np.where(flagger.isFlagged(fflags))[0])
    expected_idx = np.arange(min(idx), max(idx) + n + 1)
    assert (result_idx == expected_idx).all()


if __name__ == "__main__":

    # NOTE: PositionalFlagger is currently broken, going to fix it when needed
    # for flagger in [SimpleFlagger, PositionalFlagger, DmpFlagger]:
    for flagger in [SimpleFlagger(), DmpFlagger()]:
        test_temporalPartitioning(flagger)
        test_flagNext(flagger)
        test_missingConfig(flagger)
        test_missingVariable(flagger)
        test_assignVariable(flagger)
