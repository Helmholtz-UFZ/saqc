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
        d_saqc.outliers.flagRange("data", max=500)
        .outliers.flagRange("data", max=400)
        .outliers.flagRange("data", max=300)
    )

    # not interactive, no storing
    dummy_path = ""

    d_saqc = d_saqc.tools.plot(field="data", save_path="")
    d_saqc = d_saqc.tools.plot(
        field="data", save_path=dummy_path, plot_kwargs={"history": "valid"}, stats=True
    )
    d_saqc = d_saqc.tools.plot(
        field="data", save_path=dummy_path, plot_kwargs={"history": "all"}
    )
    d_saqc = d_saqc.tools.plot(
        field="data", save_path=dummy_path, plot_kwargs={"slice": "2000-10"}, stats=True
    )
    d_saqc.evaluate()
