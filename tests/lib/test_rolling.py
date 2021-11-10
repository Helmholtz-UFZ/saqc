import pytest

import pandas as pd
import numpy as np
from saqc.lib.rolling import customRoller

n = np.nan


def test_rolling_existence_of_attrs():
    r = pd.DataFrame().rolling(0).validate()
    c = customRoller(pd.DataFrame(), 0, min_periods=0)
    expected = [attr for attr in dir(r) if not attr.startswith("_")]
    result = [attr for attr in dir(c) if not attr.startswith("_")]
    diff = [attr for attr in expected if attr not in result]
    print(diff)
    assert len(diff) == 0


@pytest.fixture()
def data():
    # a series with a symmetrical but not regular index
    left = pd.date_range("2000", freq="1min", periods=3)
    middle = pd.date_range(left[-1], freq="1H", periods=4)
    right = pd.date_range(middle[-1], freq="1min", periods=3)
    s = pd.Series(1, index=pd.Index([*left, *middle[1:-1], *right]))
    return s


@pytest.mark.parametrize(
    "kws, expected",
    [
        (dict(window="0h", closed="neither"), [n, n, n, n, n, n, n, n]),
        (dict(window="1h", closed="neither"), [n, n, n, n, n, n, 1, 2]),
        (dict(window="2h", closed="neither"), [n, n, n, n, 1, 1, 2, 3]),
        (dict(window="3h", closed="neither"), [n, n, n, n, n, 2, 3, 4]),
        (dict(window="4h", closed="neither"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="left"), [n, n, n, n, n, n, n, n]),
        (dict(window="1h", closed="left"), [n, n, n, 1, 1, 1, 1, 2]),
        (dict(window="2h", closed="left"), [n, n, n, n, 2, 2, 2, 3]),
        (dict(window="3h", closed="left"), [n, n, n, n, n, 3, 3, 4]),
        (dict(window="4h", closed="left"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="right"), [1, 1, 1, 1, 1, 1, 1, 1]),
        (dict(window="1h", closed="right"), [n, n, n, 1, 1, 1, 2, 3]),
        (dict(window="2h", closed="right"), [n, n, n, n, 2, 2, 3, 4]),
        (dict(window="3h", closed="right"), [n, n, n, n, n, 3, 4, 5]),
        (dict(window="4h", closed="right"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="both"), [1, 1, 1, 1, 1, 1, 1, 1]),
        (dict(window="1h", closed="both"), [n, n, n, 2, 2, 2, 2, 3]),
        (dict(window="2h", closed="both"), [n, n, n, n, 3, 3, 3, 4]),
        (dict(window="3h", closed="both"), [n, n, n, n, n, 4, 4, 5]),
        (dict(window="4h", closed="both"), [n, n, n, n, n, n, n, n]),
    ],
    ids=lambda x: str(x if not isinstance(x, list) else ""),
)
def test_rolling_expand(data, kws, expected):
    expected = np.array(expected)
    result = customRoller(data, **kws, expand=False).sum()
    result = result.to_numpy()

    print()
    print(
        pd.DataFrame(
            dict(
                orig=data,
                exp=expected,
                res=result,
            ),
            index=data.index,
        )
    )
    assert np.allclose(result, expected, rtol=0, atol=0, equal_nan=True)


# left and right results are swapped
# the expected result is checked inverted, aka x[::-1]
@pytest.mark.parametrize(
    "kws, expected",
    [
        (dict(window="0h", closed="neither"), [n, n, n, n, n, n, n, n]),
        (dict(window="1h", closed="neither"), [n, n, n, n, n, n, 1, 2]),
        (dict(window="2h", closed="neither"), [n, n, n, n, 1, 1, 2, 3]),
        (dict(window="3h", closed="neither"), [n, n, n, n, n, 2, 3, 4]),
        (dict(window="4h", closed="neither"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="right"), [n, n, n, n, n, n, n, n]),
        (dict(window="1h", closed="right"), [n, n, n, 1, 1, 1, 1, 2]),
        (dict(window="2h", closed="right"), [n, n, n, n, 2, 2, 2, 3]),
        (dict(window="3h", closed="right"), [n, n, n, n, n, 3, 3, 4]),
        (dict(window="4h", closed="right"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="left"), [1, 1, 1, 1, 1, 1, 1, 1]),
        (dict(window="1h", closed="left"), [n, n, n, 1, 1, 1, 2, 3]),
        (dict(window="2h", closed="left"), [n, n, n, n, 2, 2, 3, 4]),
        (dict(window="3h", closed="left"), [n, n, n, n, n, 3, 4, 5]),
        (dict(window="4h", closed="left"), [n, n, n, n, n, n, n, n]),
        # at least #hour NANs at beginning (removed by expanding=False)
        (dict(window="0h", closed="both"), [1, 1, 1, 1, 1, 1, 1, 1]),
        (dict(window="1h", closed="both"), [n, n, n, 2, 2, 2, 2, 3]),
        (dict(window="2h", closed="both"), [n, n, n, n, 3, 3, 3, 4]),
        (dict(window="3h", closed="both"), [n, n, n, n, n, 4, 4, 5]),
        (dict(window="4h", closed="both"), [n, n, n, n, n, n, n, n]),
    ],
    ids=lambda x: str(x if not isinstance(x, list) else ""),
)
def test_rolling_expand_forward(data, kws, expected):
    expected = np.array(expected)[::-1]  # inverted
    result = customRoller(data, **kws, expand=False, forward=True).sum()
    result = result.to_numpy()

    print()
    print(
        pd.DataFrame(
            dict(
                orig=data,
                exp=expected,
                res=result,
            ),
            index=data.index,
        )
    )
    assert np.allclose(result, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("window", ["0H", "1H", "2H", "3H", "4H"])
@pytest.mark.parametrize("closed", ["both", "neither", "left", "right"])
@pytest.mark.parametrize("center", [False, True], ids=lambda x: f" center={x} ")
@pytest.mark.parametrize("forward", [False, True], ids=lambda x: f" forward={x} ")
@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "count",
        "mean",
        "median",
        "min",
        "max",
        "skew",
        "kurt",
        "cov",
        "corr",
        "sem",
        "var",
        "std",
    ],
)
def test_dtindexer(data, center, closed, window, forward, func):
    print()
    print("forward", forward)
    print("center", center)
    print("closed", closed)
    print("window", window)

    data: pd.Series

    d = data
    cl = closed
    if forward:
        d = data[::-1]
        cl = "right" if closed == "left" else "left" if closed == "right" else closed
    roller = d.rolling(
        window=window,
        closed=cl,
        center=center,
    )

    expected = getattr(roller, func)()
    if forward:
        expected = expected[::-1]

    roller = customRoller(
        obj=data,
        window=window,
        closed=closed,
        center=center,
        forward=forward,
        expand=True,
    )
    result = getattr(roller, func)()

    print()
    print(
        pd.DataFrame(
            dict(
                orig=data,
                exp=expected,
                res=result,
            ),
            index=data.index,
        )
    )

    # pandas bug
    if pd.__version__ < "1.4" and forward:
        result = result[:-1]
        expected = expected[:-1]

    # pandas bug
    # pandas insert a NaN where a valid value should be
    if (
        pd.__version__ < "1.4"
        and forward
        and func in ["sem", "var", "std"]
        and int(window[:-1]) <= 1
    ):
        pytest.skip("fails for pandas < 1.4")

    assert np.allclose(result, expected, rtol=0, atol=0, equal_nan=True)
