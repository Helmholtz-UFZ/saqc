import pytest

from saqc.lib.rolling import customRoller, Rolling
import pandas as pd
import numpy as np

FUNCTS = ['count', 'sum', 'mean', 'median', 'var', 'std', 'min', 'max', 'corr', 'cov', 'skew', 'kurt', ]

OTHA = ['apply',
        'aggregate',  # needs param func eg. func='min'
        'quantile',  # needs param quantile=0.5 (0<=q<=1)
        ]


@pytest.fixture
def data():
    return data_()


def data_():
    s1 = pd.Series(1., index=pd.date_range("1999/12", periods=12, freq='1M') + pd.Timedelta('1d'))
    s2 = pd.Series(1., index=pd.date_range('2000/05/15', periods=8, freq='1d'))
    s = pd.concat([s1, s2]).sort_index()
    s.name = 's'
    s[15] = np.nan
    return s


len_s = len(data_())


def make_num_kws():
    l = []
    n = list(range(len_s))
    for window in n:
        mp = list(range(window))
        for min_periods in [None] + mp:
            if min_periods is not None and min_periods > window:
                continue
            for center in [False, True]:
                l.append(dict(window=window, min_periods=min_periods, center=center))
    return l


def make_dt_kws():
    l = []
    n = [0, 1, 2, 10, 32, 70, 120]
    mp = list(range(len_s))
    for closed in ['right', 'both', 'neither', 'left']:
        for window in n:
            for min_periods in [None] + mp:
                l.append(dict(window=f'{window}d', min_periods=min_periods, closed=closed))
    return l


def check_series(result, expected):
    if not (result.isna() == expected.isna()).all():
        return False
    result = result.dropna()
    expected = expected.dropna()
    if not (result == expected).all():
        return False
    return True


def print_diff(s, result, expected):
    df = pd.DataFrame()
    df['s'] = s
    df['exp'] = expected
    df['res'] = result
    print(df)


def call_rolling_function(roller, func):
    if isinstance(func, str):
        return getattr(roller, func)()
    else:
        return getattr(roller, 'apply')(func)


@pytest.mark.parametrize("kws", make_dt_kws(), ids=lambda x: str(x))
@pytest.mark.parametrize("func", FUNCTS)
def test_pandas_conform_dt(data, kws, func):
    s = data
    try:
        expR = s.rolling(**kws)
        expected = call_rolling_function(expR, func)
    except Exception as e0:
        # pandas failed, so we should also fail
        try:
            resR = customRoller(s, **kws)
            result = call_rolling_function(resR, func)
        except Exception as e1:
            assert type(e0) == type(e1)
            return
        assert False, 'pandas faild, but we succeed'

    resR = customRoller(s, **kws)
    result = call_rolling_function(resR, func)
    success = check_series(result, expected)
    if success:
        return
    print_diff(s, result, expected)
    assert False


@pytest.mark.parametrize("kws", make_num_kws(), ids=lambda x: str(x))
@pytest.mark.parametrize("func", FUNCTS)
def test_pandas_conform_num(data, kws, func):
    s = data
    try:
        expR = s.rolling(**kws)
        expected = call_rolling_function(expR, func)
    except Exception as e0:
        # pandas failed, so we should also fail
        try:
            resR = customRoller(s, **kws)
            result = call_rolling_function(resR, func)
        except Exception as e1:
            assert type(e0) == type(e1)
            return
        assert False, 'pandas faild, but we succeed'

    resR = customRoller(s, **kws)
    result = call_rolling_function(resR, func)
    success = check_series(result, expected)
    if success:
        return
    print_diff(s, result, expected)
    assert False


@pytest.mark.parametrize("kws", make_dt_kws(), ids=lambda x: str(x))
@pytest.mark.parametrize("func", FUNCTS)
def test_forward_dt(data, kws, func):
    s = data
    try:
        expR = pd.Series(reversed(s), reversed(s.index)).rolling(**kws)
        expected = call_rolling_function(expR, func)[::-1]
    except Exception as e0:
        # pandas failed, so we should also fail
        try:
            resR = customRoller(s, forward=True, **kws)
            result = call_rolling_function(resR, func)
        except Exception as e1:
            assert type(e0) == type(e1)
            return
        assert False, 'pandas faild, but we succeed'

    resR = customRoller(s, forward=True, **kws)
    result = call_rolling_function(resR, func)
    success = check_series(result, expected)
    if success:
        return
    print_diff(s, result, expected)
    assert False


@pytest.mark.parametrize("kws", make_num_kws(), ids=lambda x: str(x))
@pytest.mark.parametrize("func", FUNCTS)
def test_forward_num(data, kws, func):
    s = data
    try:
        expR = pd.Series(reversed(s), reversed(s.index)).rolling(**kws)
        expected = call_rolling_function(expR, func)[::-1]
    except Exception as e0:
        # pandas failed, so we should also fail
        try:
            resR = customRoller(s, forward=True, **kws)
            result = call_rolling_function(resR, func)
        except Exception as e1:
            assert type(e0) == type(e1)
            return
        assert False, 'pandas faild, but we succeed'

    resR = customRoller(s, forward=True, **kws)
    result = call_rolling_function(resR, func)
    success = check_series(result, expected)
    if success:
        return
    print_diff(s, result, expected)
    assert False


def dt_center_kws():
    l = []
    for window in range(2, 10, 2):
        for min_periods in range(1, window + 1):
            l.append(dict(window=window, min_periods=min_periods))
    return l


@pytest.mark.parametrize("kws", dt_center_kws(), ids=lambda x: str(x))
def test_centering_w_dtindex(kws):
    print(kws)
    s = pd.Series(0., index=pd.date_range("2000", periods=10, freq='1H'))
    s[4:7] = 1

    w = kws.pop('window')
    mp = kws.pop('min_periods')

    pd_kw = dict(window=w, center=True, min_periods=mp)
    our_kw = dict(window=f'{w}h', center=True, closed='both', min_periods=mp)
    expected = s.rolling(**pd_kw).sum()
    result = customRoller(s, **our_kw).sum()
    success = check_series(result, expected)
    if not success:
        print_diff(s, result, expected)
        assert False

    w -= 1
    mp -= 1
    pd_kw = dict(window=w, center=True, min_periods=mp)
    our_kw = dict(window=f'{w}h', center=True, closed='neither', min_periods=mp)
    expected = s.rolling(**pd_kw).sum()
    result = customRoller(s, **our_kw).sum()
    success = check_series(result, expected)
    if not success:
        print_diff(s, result, expected)
        assert False
