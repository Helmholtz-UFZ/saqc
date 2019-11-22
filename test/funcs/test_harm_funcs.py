#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from saqc.flagger.dmpflagger import DmpFlagger
from saqc.flagger.simpleflagger import SimpleFlagger
from saqc.flagger.continuousflagger import ContinuousBaseFlagger

from saqc.funcs.harm_functions import harm_wrapper

from saqc.lib.tools import getPandasData
from saqc.funcs.functions import flagMissing


TESTFLAGGERS = [
    DmpFlagger(),
    SimpleFlagger()
    #ContinuousBaseFlagger()
    ]

RESHAPERS = [
    'nearest_shift',
    'fshift',
    'bshift'
]

COFLAGGING = [
    False,
    True
]

SETSHIFTCOMMENT = [
    False,
    True
]

INTERPOLATIONS = [
    'fshift',
    'bshift',
    'nearest_shift',
    'nearest_agg',
    'bagg'
]

FREQS = [
    '15min',
    '30min'
]

@pytest.fixture(scope='module')
def data():
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 01:00:00', freq='15min')
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp('2011-01-01 00:30:00'))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name='data')
    # good to have some nan:
    dat[-3] = np.nan
    data = dat.to_frame()
    return data

@pytest.fixture(scope='module')
def multi_data():
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 01:00:00', freq='15min')
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp('2011-01-01 00:30:00'))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name='data')
    # good to have some nan:
    dat[-3] = np.nan
    data = dat.to_frame()
    data.index = data.index.shift(1, '2min')
    dat2 = data.copy()
    dat2.index = dat2.index.shift(1, '17min')
    dat2.rename(columns={'data': 'data2'}, inplace=True)
    dat3 = data.copy()
    dat3.index = dat3.index.shift(1, '1h')
    dat3.rename(columns={'data': 'data3'}, inplace=True)
    dat3.drop(dat3.index[2:-2], inplace=True)
    # merge
    data = pd.merge(data, dat2, how='outer', left_index=True, right_index=True)
    data = pd.merge(data, dat3, how='outer', left_index=True, right_index=True)
    return data


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('reshaper', RESHAPERS)
@pytest.mark.parametrize('co_flagging', COFLAGGING)
def test_harm_single_var_intermediate_flagging(data, flagger, reshaper, co_flagging):

    flags = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flags.copy()
    freq = '15min'

    # harmonize data:
    data, flags = harm_wrapper()(data, flags, 'data', flagger, freq, 'time', reshaper)

    # flag something bad
    flags = flagger.setFlags(flags, 'data', loc=flags.index[3:4])
    data, flags = harm_wrapper(harm=False)(data, flags, 'data', flagger, co_flagging=co_flagging)

    if reshaper is 'nearest_shift':
        if co_flagging is True:
            assert flagger.isFlagged(flags.loc[flags.index[3:7]]).squeeze().all()
            assert (~flagger.isFlagged(flags.loc[flags.index[0:3]]).squeeze()).all()
            assert (~flagger.isFlagged(flags.loc[flags.index[7:]]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged(flags).squeeze() ==
                    [False, False, False, False, True, False, True, False, False]).all()
    if reshaper is 'bshift':
        if co_flagging is True:
            assert flagger.isFlagged(flags.loc[flags.index[5:7]]).squeeze().all()
            assert (~flagger.isFlagged(flags.loc[flags.index[0:5]]).squeeze()).all()
            assert (~flagger.isFlagged(flags.loc[flags.index[7:]]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged(flags).squeeze() ==
                    [False, False, False, False, False, True, True, False, False]).all()
    if reshaper is 'fshift':
        if co_flagging is True:
            assert flagger.isFlagged(flags.loc[flags.index[3:5]]).squeeze().all()
            assert flagger.isFlagged(flags.loc[flags.index[6:7]]).squeeze().all()
            assert (~flagger.isFlagged(flags.loc[flags.index[0:3]]).squeeze()).all()
            assert (~flagger.isFlagged(flags.loc[flags.index[7:]]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged(flags).squeeze() ==
                    [False, False, False, False, True, False, True, False, False]).all()

    assert pre_data.equals(data)
    assert len(data) == len(flags)
    assert (pre_flags.index == flags.index).all()

@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('interpolation', INTERPOLATIONS)
@pytest.mark.parametrize('freq', FREQS)
def test_harm_single_var_interpolations(data, flagger, interpolation, freq):
    flags = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flags.copy()

    harm_start = data.index[0].floor(freq=freq)
    harm_end = data.index[-1].ceil(freq=freq)
    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    data, flags = harm_wrapper()(data, flags, 'data', flagger, freq, interpolation, 'fshift',
                                 reshape_shift_comment=False, inter_agg=np.sum)

    if interpolation is 'fshift':
        if freq == '15min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -37.5, -25.0, 0.0, 37.5, 50.0]}, index=test_index))
        if freq == '30min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -37.5, 0.0, 50.0]}, index=test_index))
    if interpolation is 'bshift':
        if freq == '15min':
            assert data.equals(pd.DataFrame({'data': [-50.0, -37.5, -25.0, 12.5, 37.5, 50.0]}, index=test_index))
        if freq == '30min':
            assert data.equals(pd.DataFrame({'data': [-50.0, -37.5, 12.5, 50.0]}, index=test_index))
    if interpolation is 'nearest_shift':
        if freq == '15min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -37.5, -25.0, 12.5, 37.5, 50.0]}, index=test_index))
        if freq == '30min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -37.5, 12.5, 50.0]}, index=test_index))
    if interpolation is 'nearest_agg':
        if freq == '15min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -87.5, -25.0, 0.0, 37.5, 50.0]}, index=test_index))
        if freq == '30min':
            assert data.equals(pd.DataFrame({'data': [np.nan, -87.5, -25.0, 87.5]}, index=test_index))
    if interpolation is 'bagg':
        if freq == '15min':
            assert data.equals(pd.DataFrame({'data': [-50.0, -37.5, -37.5, 12.5, 37.5, 50.0]}, index=test_index))
        if freq == '30min':
            assert data.equals(pd.DataFrame({'data': [-50.0, -75.0, 50.0, 50.0]}, index=test_index))

    data, flags = harm_wrapper(harm=False)(data, flags, 'data', flagger, co_flagging=True)

    data, flags = harm_wrapper(harm=False)(data, flags, 'data', flagger, co_flagging=True)

    assert pre_data.equals(data)
    assert len(data) == len(flags)
    assert (pre_flags.index == flags.index).all()


@pytest.mark.parametrize('flagger', TESTFLAGGERS)
@pytest.mark.parametrize('shift_comment', SETSHIFTCOMMENT)
def test_multivariat_harmonization(multi_data, flagger, shift_comment):
    flags = flagger.initFlags(multi_data)
    # for comparison
    pre_data = multi_data.copy()
    pre_flags = flags.copy()
    freq = '15min'

    harm_start = multi_data.index[0].floor(freq=freq)
    harm_end = multi_data.index[-1].ceil(freq=freq)
    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    # harm:
    multi_data, flags = harm_wrapper()(multi_data, flags, 'data', flagger, freq, 'time', 'nearest_shift',
                                       reshape_shift_comment=shift_comment)
    multi_data, flags = harm_wrapper()(multi_data, flags, 'data2', flagger, freq, 'bagg', 'bshift', inter_agg=sum,
                                       reshape_agg=max, reshape_shift_comment=shift_comment)
    multi_data, flags = harm_wrapper()(multi_data, flags, 'data3', flagger, freq, 'fshift', 'fshift',
                                       reshape_shift_comment=shift_comment)

   # pdb.set_trace()
    assert multi_data.index.equals(test_index)
    assert pd.Timedelta(pd.infer_freq(multi_data.index)) == pd.Timedelta(freq)

    multi_data, flags = harm_wrapper(harm=False)(multi_data, flags, 'data3', flagger, co_flagging=False)
    multi_data, flags = harm_wrapper(harm=False)(multi_data, flags, 'data2', flagger, co_flagging=True)
    multi_data, flags = harm_wrapper(harm=False)(multi_data, flags, 'data', flagger, co_flagging=True)

    assert pre_data.equals(multi_data[pre_data.columns.to_list()])
    assert len(multi_data) == len(flags)
    assert (pre_flags.index == flags.index).all()

if __name__ == "__main__":
    data = data()
    multi_data = multi_data()
    flagger = SimpleFlagger()
    #flagger = DmpFlagger()
    flagger = ContinuousBaseFlagger()


    test_multivariat_harmonization(multi_data,  ContinuousBaseFlagger(), shift_comment=True)
    test_multivariat_harmonization(data, DmpFlagger())
    test_multivariat_harmonization(data, SimpleFlagger())







