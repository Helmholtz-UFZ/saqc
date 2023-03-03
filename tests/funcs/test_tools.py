# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import saqc
from saqc.core import DictOfSeries


@pytest.mark.slow
def test_makeFig(tmp_path):
    # just testing for no errors to occure...
    data = DictOfSeries(
        data=pd.Series(
            np.linspace(0, 1000, 1000),
            pd.date_range("2000", "2001", periods=1000),
        )
    )
    d_saqc = saqc.SaQC(data)
    d_saqc = (
        d_saqc.flagRange("data", max=500)
        .flagRange("data", max=400)
        .flagRange("data", max=300)
    )

    # not interactive, no storing
    outfile = str(Path(tmp_path, "test.png"))  # the filesystem's temp dir

    d_saqc = d_saqc.plot(field="data", path=outfile)
    d_saqc = d_saqc.plot(field="data", path=outfile, history="valid", stats=True)
    d_saqc = d_saqc.plot(field="data", path=outfile, history="complete")
    d_saqc = d_saqc.plot(
        field="data", path=outfile, ax_kwargs={"ylabel": "data is data"}, stats=True
    )
