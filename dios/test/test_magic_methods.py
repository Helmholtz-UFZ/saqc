from .test_setup import *


def test__len__(datetime_series, maxlen=10):
    dios = DictOfSeries()
    assert len(dios) == 0

    for i in range(maxlen):
        dios[f"c{i}"] = datetime_series.copy()
        assert len(dios) == i + 1

    for i in reversed(range(maxlen)):
        assert len(dios) == i + 1
        del dios[f"c{i}"]

    assert len(dios) == 0
