#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import dios

from saqc.funcs.proc_functions import (
    proc_interpolateMissing,
    proc_resample,
    proc_transform
)

from test.common import TESTFLAGGER

@pytest.mark.parametrize("flagger", TESTFLAGGER)
def test_interpolateMissing(course_1, flagger):
    data, *_ = course_1(periods=100)
    data[1] = np.nan
    data[]
