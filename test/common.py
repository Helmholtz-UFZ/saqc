#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import re

import numpy as np
import pandas as pd

from saqc.core.core import prepareConfig, readConfig
from saqc.flagger import (
    ContinuousFlagger,
    CategoricalFlagger,
    SimpleFlagger,
    DmpFlagger,
)


TESTNODATA = (np.nan, -9999)


TESTFLAGGER = (
    CategoricalFlagger(["NIL", "GOOD", "BAD"]),
    SimpleFlagger(),
    DmpFlagger(),
    ContinuousFlagger(),
)


def dummyRegisterFunc(data, field, flagger, kwarg, **kwargs):
    return data, flagger


def initData(cols=2, start_date="2017-01-01", end_date="2017-12-31", freq="1h"):
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {}
    dummy = np.arange(len(dates))
    for col in range(1, cols + 1):
        data[f"var{col}"] = dummy * (col)
    return pd.DataFrame(data, index=dates)


def initMetaString(metastring, data):
    cleaned = re.sub(
        r"\s*,\s*", r",", re.sub(r"\|", r";", re.sub(r"\n[ \t]+", r"\n", metastring))
    )
    fobj = io.StringIO(cleaned)
    meta = prepareConfig(readConfig(fobj), data)
    fobj.seek(0)
    return fobj, meta


def _getKeys(metadict):
    keys = list(metadict[0].keys())
    for row in metadict[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    return keys


def initMetaDict(config_dict, data):
    df = pd.DataFrame(config_dict)[_getKeys(config_dict)]
    meta = prepareConfig(df, data)
    fobj = io.StringIO()
    meta.to_csv(fobj, index=False, sep=";")
    fobj.seek(0)
    return fobj, meta
