#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from core import flaggingRunner
from config import Fields
from flagger import SimpleFlagger
from .testfuncs import initData


def initMeta(data):
    dates = data.index
    variables = data.columns
    randg = np.random.randint
    start_dates = [dates[randg(0, (len(dates)//2)-1)] for _ in variables]
    end_dates = [dates[randg(len(dates)//2, len(dates) - 1 )] for _ in variables]
    tests = ["generic; func: abs(this) + 1 > 0"] * len(variables)
    return pd.DataFrame({Fields.VARNAME: data.columns,
                         Fields.STARTDATE: start_dates,
                         Fields.ENDDATE: end_dates,
                         Fields.FLAGS: tests})


def testTemporalPartitioning():

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


if __name__ == "__main__":
    testTemporalPartitioning()
