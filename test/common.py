#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import re

import numpy as np
import pandas as pd

from saqc.core.core import prepareMeta, readMeta
from saqc.flagger import SimpleFlagger, DmpFlagger


TESTFLAGGER = (SimpleFlagger(), DmpFlagger())


def initData(cols=2, start_date="2017-01-01", end_date="2017-12-31", freq="1h"):
    dates = pd.date_range(start="2017-01-01", end="2017-12-31", freq="1h")
    data = {}
    dummy = np.arange(len(dates))
    for col in range(1, cols+1):
        data[f"var{col}"] = dummy*(col)
    return pd.DataFrame(data, index=dates)


def initMeta(metastring, data):
    cleaned = re.sub(r"\s*,\s*", r",",
                     re.sub(r"\|", r",",
                            re.sub(r"\n[ \t]+", r"\n",
                                   metastring)))
    fobj = io.StringIO(cleaned)
    meta = prepareMeta(readMeta(fobj), data)
    fobj.seek(0)
    return fobj, meta


def initMetaDict(metadict, data):
    meta = prepareMeta(pd.DataFrame(metadict), data)
    fobj = io.StringIO()
    meta.to_csv(fobj)
    fobj.seek(0)
    return fobj, meta
