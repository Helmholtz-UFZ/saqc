#!/usr/bin/env python
import pytest
from .test_setup import *

import pandas as pd
from pandas.core.dtypes.common import is_dict_like, is_nested_list_like
import numpy as np
from copy import deepcopy

__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2018, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"


arr = np.random.rand(8)
TESTDATA = [
    None,  # empty  # 0
    [1],  # 1
    arr.copy(),  # 2
    np.array([arr.copy(), arr.copy(), arr.copy()]),  # 3 - nested list
    range(4),  # 4
    dict(a=arr.copy(), b=arr.copy()),  # 5 dict
    pd.DataFrame(dict(a=arr.copy(), b=arr.copy())),  # 6 df
]


@pytest.mark.parametrize("data", TESTDATA)
@pytest.mark.parametrize("with_column_param", [False, True])
def test_dios_create(data, with_column_param):

    data_copy0 = deepcopy(data)
    data_copy1 = deepcopy(data)

    # create columns list
    if with_column_param:
        df = pd.DataFrame(data=data_copy0)
        col = [f"new_{c}" for c in df]
    else:
        col = None

    if is_nested_list_like(data):
        # giving nested lists, work different between df and dios
        data_copy1 = data_copy1.transpose()

    df = pd.DataFrame(data=data_copy0, columns=col)
    dios = DictOfSeries(data=data_copy1, columns=col)

    assert dios.columns.equals(df.columns)

    eq, msg = dios_eq_df(dios, df, with_msg=True)
    assert eq, msg
