#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import pandas as pd

from dios import dios
from test.common import TESTFLAGGER

from saqc.funcs.data_modelling import (
    modelling_polyFit
)
import numpy.polynomial.polynomial as poly


@pytest.mark.parametrize("flagger", TESTFLAGGER)
@pytest.mark.parametrize("dat", [pytest.lazy_fixture("course_2")])
def test_modelling_polyFit_forRegular(dat, flagger):
    data, _ = dat(freq='10min', periods=100, initial_level=0, final_level=100, out_val=-100)
    # add some nice sine distortion
    data += np.sin(np.arange(0, len(data)))
    data = dios.DictOfSeries(data)
    flagger = flagger.initFlags(data)

    result1, _ = modelling_polyFit(data, 'data', flagger, 11, numba=False)
    result2, _ = modelling_polyFit(data, 'data', flagger, 11, numba=True)
    result3, _ = modelling_polyfit(data, 'data', flagger, '2h', numba=False)
    result4, _ = modelling_polyfit(data, 'data', flagger, '2h', numba=True)