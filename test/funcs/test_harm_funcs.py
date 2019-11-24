#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import pandas as pd

from saqc.funcs.harm_functions import harm_wrapper

from test.common import TESTFLAGGER


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


@pytest.fixture
def data():
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 01:00:00', freq='15min')
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp('2011-01-01 00:30:00'))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name='data')
    # good to have some nan
    dat[-3] = np.nan
    data = dat.to_frame()
    return data


@pytest.fixture
def multi_data():
    index = pd.date_range(start='1.1.2011 00:00:00', end='1.1.2011 01:00:00', freq='15min')
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 29, 0))
    index = index.insert(2, pd.Timestamp(2011, 1, 1, 0, 28, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 32, 0))
    index = index.insert(5, pd.Timestamp(2011, 1, 1, 0, 31, 0))
    index = index.insert(0, pd.Timestamp(2010, 12, 31, 23, 57, 0))
    index = index.drop(pd.Timestamp('2011-01-01 00:30:00'))
    dat = pd.Series(np.linspace(-50, 50, index.size), index=index, name='data')
    # good to have some nan
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


@pytest.mark.parametrize('flagger', TESTFLAGGER)
@pytest.mark.parametrize('reshaper', RESHAPERS)
@pytest.mark.parametrize('co_flagging', COFLAGGING)
def test_harm_single_var_intermediate_flagging(data, flagger, reshaper, co_flagging):

    flagger = flagger.initFlags(data)
    # flags = flagger.initFlags(data)
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flagger.getFlags()
    freq = '15min'

    # harmonize data:
    data, flagger = harm_wrapper()(data, 'data', flagger, freq, 'time', reshaper)

    # flag something bad
    flagger = flagger.setFlags('data', loc=data.index[3:4])
    data, flagger = harm_wrapper(harm=False)\
        (data, 'data', flagger, co_flagging=co_flagging)

    if reshaper is 'nearest_shift':
        if co_flagging is True:
            assert flagger.isFlagged(loc=data.index[3:7]).squeeze().all()
            assert (~flagger.isFlagged(loc=data.index[0:3]).squeeze()).all()
            assert (~flagger.isFlagged(loc=data.index[7:]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged().squeeze() ==
                    [False, False, False, False, True, False, True, False, False]).all()
    if reshaper is 'bshift':
        if co_flagging is True:
            assert flagger.isFlagged(loc=data.index[5:7]).squeeze().all()
            assert (~flagger.isFlagged(loc=data.index[0:5]).squeeze()).all()
            assert (~flagger.isFlagged(loc=data.index[7:]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged().squeeze() ==
                    [False, False, False, False, False, True, True, False, False]).all()
    if reshaper is 'fshift':
        if co_flagging is True:
            assert flagger.isFlagged(loc=data.index[3:5]).squeeze().all()
            assert flagger.isFlagged(loc=data.index[6:7]).squeeze().all()
            assert (~flagger.isFlagged(loc=data.index[0:3]).squeeze()).all()
            assert (~flagger.isFlagged(loc=data.index[7:]).squeeze()).all()
        if co_flagging is False:
            assert (flagger.isFlagged().squeeze() ==
                    [False, False, False, False, True, False, True, False, False]).all()

    flags = flagger.getFlags()
    assert pre_data.equals(data)
    assert len(data) == len(flags)
    assert (pre_flags.index == flags.index).all()

@pytest.mark.parametrize('flagger', TESTFLAGGER)
@pytest.mark.parametrize('interpolation', INTERPOLATIONS)
@pytest.mark.parametrize('freq', FREQS)
def test_harm_single_var_interpolations(data, flagger, interpolation, freq):
    flagger = flagger.initFlags(data)
    flags = flagger.getFlags()
    # make pre harm copies:
    pre_data = data.copy()
    pre_flags = flags.copy()

    harm_start = data.index[0].floor(freq=freq)
    harm_end = data.index[-1].ceil(freq=freq)
    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    data, flagger = harm_wrapper()\
        (data, 'data', flagger, freq, interpolation, 'fshift',
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

    data, flagger = harm_wrapper(harm=False)(data, 'data', flagger, co_flagging=True)

    data, flagger = harm_wrapper(harm=False)(data, 'data', flagger, co_flagging=True)
    flags = flagger.getFlags()

    assert pre_data.equals(data)
    assert len(data) == len(flags)
    assert (pre_flags.index == flags.index).all()


@pytest.mark.parametrize('flagger', TESTFLAGGER)
@pytest.mark.parametrize('shift_comment', SETSHIFTCOMMENT)
def test_multivariat_harmonization(multi_data, flagger, shift_comment):
    flagger = flagger.initFlags(multi_data)
    flags = flagger.getFlags()
    # for comparison
    pre_data = multi_data.copy()
    pre_flags = flags.copy()
    freq = '15min'

    harm_start = multi_data.index[0].floor(freq=freq)
    harm_end = multi_data.index[-1].ceil(freq=freq)
    test_index = pd.date_range(start=harm_start, end=harm_end, freq=freq)
    # harm:
    harmonizer = harm_wrapper()
    deharmonizer = harm_wrapper(harm=False)
    multi_data, flagger = harmonizer(
        multi_data, 'data', flagger,
        freq, 'time', 'nearest_shift',
        reshape_shift_comment=shift_comment)

    multi_data, flagger = harmonizer(
        multi_data, 'data2', flagger,
        freq, 'bagg', 'bshift',
        inter_agg=sum,
        reshape_agg=max,
        reshape_shift_comment=shift_comment)

    multi_data, flagger= harmonizer(
        multi_data, 'data3', flagger,
        freq, 'fshift', 'fshift',
         reshape_shift_comment=shift_comment)

    assert multi_data.index.equals(test_index)
    assert pd.Timedelta(pd.infer_freq(multi_data.index)) == pd.Timedelta(freq)

    multi_data, flagger = deharmonizer(
        multi_data, 'data3', flagger,
        co_flagging=False)
    multi_data, flagger = deharmonizer(
        multi_data, 'data2', flagger,
        co_flagging=True)
    multi_data, flagger = deharmonizer(
        multi_data, 'data', flagger,
        co_flagging=True)

    flags = flagger.getFlags()
    assert pre_data.equals(multi_data[pre_data.columns.to_list()])
    assert len(multi_data) == len(flags)
    assert (pre_flags.index == flags.index).all()
