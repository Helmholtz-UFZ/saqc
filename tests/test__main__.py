#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
import pytest


def test_unknownFileExtention():
    from saqc.__main__ import readData, setupIO, writeData

    reader, writer = setupIO(np.nan)
    with pytest.raises(ValueError):
        readData(reader, "foo.unknown")
    with pytest.raises(ValueError):
        writeData(reader, pd.DataFrame(), "foo.unknown")
