#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from core import flaggingRunner, flagNext
from config import Fields
from flagger import SimpleFlagger, DmpFlagger
from .testfuncs import initData


def initMeta(data):
    dates = data.index
    variables = data.columns
    randg = np.random.randint
    start_dates = [dates[randg(0, (len(dates)//2)-1)] for _ in variables]
    end_dates = [dates[randg(len(dates)//2, len(dates) - 1 )] for _ in variables]
    tests = ["generic, {func: abs(this) + 1 > 0}"] * len(variables)
    return pd.DataFrame({Fields.VARNAME: data.columns,
                         Fields.STARTDATE: start_dates,
                         Fields.ENDDATE: end_dates,
                         Fields.FLAGS: tests})


def test_temporalPartitioning():

    data = initData()
    meta = initMeta(data)
    flagger = SimpleFlagger()
    pdata, pflags = flaggingRunner(meta, flagger, data)

    fields = [Fields.VARNAME, Fields.STARTDATE, Fields.ENDDATE]
    for _, row in meta.iterrows():
        vname, start_date, end_date = row[fields]
        fchunk = pflags[vname].dropna()
        assert fchunk.index.min() == start_date, "different start dates"
        assert fchunk.index.max() == end_date, "different end dates"


def test_flagNextFill():
    flagger = SimpleFlagger()
    data = initData().iloc[:, 1]
    flags = flagger.emptyFlags(data)

    idx = [0, 1, 2]
    flags.iloc[idx] = flagger.setFlag(flags.iloc[idx])

    n = 4
    fflags = flagNext(flagger, flags.copy(), 4)
    result_idx = np.unique(np.where(pd.notnull(fflags))[0])
    expected_idx = np.arange(min(idx), max(idx) + n + 1)
    assert (result_idx == expected_idx).all()


def test_flagNextOverwrite():
    flagger = SimpleFlagger()
    data = initData().iloc[:, 0]
    flags = flagger.emptyFlags(data)

    flags.iloc[0::3] = flagger.setFlag(flags.iloc[0::3], 1)
    flags.iloc[2::3] = flagger.setFlag(flags.iloc[2::3], 2)

    fflags = flagNext(flagger, flags.copy().iloc[:], 4)
    assert ((fflags.values[pd.isnull(flags)] == 1).all(axis=None))


def test_flagNextMulticolumn():
    flagger = DmpFlagger()
    data = initData().iloc[:, 0]
    flags = flagger.emptyFlags(data)

    flags.iloc[0::3] = flagger.setFlag(flags.iloc[0::3], "DOUBTFUL")
    flags.iloc[2::3] = flagger.setFlag(flags.iloc[2::3], "BAD")

    fflags = flagNext(flagger, flags.copy(), 4)
    assert ((fflags.values[pd.isnull(flags)] == 1).all(axis=None))


if __name__ == "__main__":
    test_temporalPartitioning()
    test_flagNextFill()
    test_flagNextOverwrite()
    test_flagNextMulticolumn()
