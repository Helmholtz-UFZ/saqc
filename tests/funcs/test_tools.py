import pytest

from saqc.lib.plotting import makeFig
import pandas as pd
import numpy as np
import saqc
import dios


def test_makeFig():
    # just testing for no errors to occure...
    data = dios.DictOfSeries(
        pd.Series(
            np.linspace(0, 1000, 1000),
            pd.date_range("2000", "2001", periods=1000),
            name="data",
        )
    )
    d_saqc = saqc.SaQC(data)
    d_saqc = (
        d_saqc.flagRange("data", max=500)
        .flagRange("data", max=400)
        .flagRange("data", max=300)
    )

    # not interactive, no storing
    dummy_path = ""

    d_saqc = d_saqc.plot(field="data", path="")
    d_saqc = d_saqc.plot(field="data", path=dummy_path, history="valid", stats=True)
    d_saqc = d_saqc.plot(field="data", path=dummy_path, history="complete")
    d_saqc = d_saqc.plot(
        field="data", path=dummy_path, ax_kwargs={"ylabel": "data is data"}, stats=True
    )
