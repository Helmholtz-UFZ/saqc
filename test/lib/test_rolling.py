import pytest

from saqc.lib.rolling import customRoller
import pandas as pd
import numpy as np


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
    for window in range(len_s + 2):
        for min_periods in [None] + list(range(window + 1)):
            for center in [False, True]:
                for closed in [None] + ['left', 'right', 'both', 'neither']:
                    l.append(dict(window=window, min_periods=min_periods, center=center, closed=closed))
    return l


def make_dt_kws():
    l = []
    for closed in [None] + ['right', 'both', 'neither', 'left']:
        for window in range(1, len_s + 3):
            for min_periods in [None] + list(range(window + 1)):
                for win in [f'{window}d', f'{window * 31}d']:
                    l.append(dict(window=win, min_periods=min_periods, closed=closed))
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


def runtest_for_kw_combi(s, kws):
    print(kws)
    forward = kws.pop('forward', False)
    if forward:
        result = customRoller(s, forward=True, **kws).sum()
        expected = pd.Series(reversed(s), reversed(s.index)).rolling(**kws).sum()[::-1]

        success = check_series(result, expected)
        if not success:
            print_diff(s, result, expected)
            assert False, f"forward=True !! {kws}"
    else:
        result = customRoller(s, **kws).sum()
        expected = s.rolling(**kws).sum()

        success = check_series(result, expected)
        if not success:
            print_diff(s, result, expected)
            assert False


@pytest.mark.parametrize("kws", make_num_kws())
def test_pandas_conform_num(data, kws):
    runtest_for_kw_combi(data, kws)


@pytest.mark.parametrize("kws", make_dt_kws())
def test_pandas_conform_dt(data, kws):
    runtest_for_kw_combi(data, kws)


@pytest.mark.parametrize("kws", make_num_kws())
def test_forward_num(data, kws):
    kws.update(forward=True)
    runtest_for_kw_combi(data, kws)


@pytest.mark.parametrize("kws", make_dt_kws())
def test_forward_dt(data, kws):
    kws.update(forward=True)
    runtest_for_kw_combi(data, kws)


def dt_center_kws():
    l = []
    for window in range(2, 10, 2):
        for min_periods in range(1, window + 1):
            l.append(dict(window=window, min_periods=min_periods))
    return l


@pytest.mark.parametrize("kws", dt_center_kws())
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
