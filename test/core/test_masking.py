#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pandas as pd

from hypothesis import given, settings
from hypothesis.strategies import (
    sampled_from,
    composite,
    sampled_from,
)

from saqc.core.core import _maskData, _unmaskData

from test.common import dataFieldFlagger, MAX_EXAMPLES


logging.disable(logging.CRITICAL)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flagger=dataFieldFlagger())
def test_maskingMasksData(data_field_flagger):
    """
    test if flagged values are replaced by np.nan
    """
    data_in, field, flagger = data_field_flagger
    data_masked, _ = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)
    assert data_masked.aloc[flagger.isFlagged(flagger.BAD)].isna().all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flagger=dataFieldFlagger())
def test_dataMutationPreventsUnmasking(data_field_flagger):
    """ test if (un)masking works as expected on data-changes.

    if `data` is mutated after `_maskData`, `_unmaskData` should be a no-op
    """
    filler = -9999

    data_in, field, flagger = data_field_flagger
    data_masked, mask = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)
    data_masked[field] = filler
    data_out = _unmaskData(data_in, mask, data_masked, flagger, to_mask=flagger.BAD)
    assert (data_out[field] == filler).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flagger=dataFieldFlagger())
def test_flaggerMutationPreventsUnmasking(data_field_flagger):
    """ test if (un)masking works as expected on flagger-changes.

    if `flagger` is mutated after `_maskData`, `_unmaskData` should be a no-op
    """
    data_in, field, flagger = data_field_flagger
    data_masked, mask = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)
    flagger = flagger.setFlags(field, flag=flagger.UNFLAGGED, force=True)
    data_out = _unmaskData(data_in, mask, data_masked, flagger, to_mask=flagger.BAD)
    assert (data_out.loc[flagger.isFlagged(field, flag=flagger.BAD), field].isna()).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flagger=dataFieldFlagger())
def test_reshapingPreventsUnmasking(data_field_flagger):
    """ test if (un)masking works as expected on index-changes.

    If the index of data (and flags) change in the func, the unmasking,
    should not reapply original data, instead take the new data (and flags) as is.
    """

    filler = -1111

    data_in, field, flagger = data_field_flagger
    data_masked, mask = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)

    # mutate indexes of `data` and `flagger`
    index = data_masked[field].index.to_series()
    index.iloc[-len(data_masked[field])//2:] += pd.Timedelta("7.5Min")
    data_masked[field] = pd.Series(data=filler, index=index)
    flags = flagger.getFlags()
    flags[field] = pd.Series(data=flags[field].values, index=index)
    flagger = flagger.initFlags(flags=flags)

    data_out = _unmaskData(data_in, mask, data_masked, flagger, to_mask=flagger.BAD)
    assert (data_out[field] == filler).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flagger=dataFieldFlagger())
def test_unmaskingInvertsMasking(data_field_flagger):
    """
    unmasking data should invert the masking
    """
    data_in, field, flagger = data_field_flagger
    data_masked, mask = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)
    data_out = _unmaskData(data_in, mask, data_masked, flagger, to_mask=flagger.BAD)
    assert data_in.to_df().equals(data_out.to_df())


# @settings(max_examples=MAX_EXAMPLES, deadline=None)
# @given(data_field_flagger=dataFieldFlagger(), func_kwargs=flagFuncsKwargs())
# def test_maskingPreservesData(data_field_flagger, func_kwargs):
#     """
#     no mutations on pre-flagged data

#     calling a function on pre-flagged data should yield the same
#     behavior as calling this function on data where the flagged values
#     are removed
#     """

#     data_in, field, flagger = data_field_flagger

#     data_masked, mask = _maskData(data_in, flagger, columns=[field], to_mask=flagger.BAD)
#     func, kwargs = func_kwargs
#     data_masked, _ = func(data_masked, field, flagger, **kwargs)
#     data_out = _unmaskData(data_in, mask, data_masked, flagger, to_mask=flagger.BAD)

#     flags_in = flagger.isFlagged(flag=flagger.BAD)
#     assert data_in.aloc[flags_in].equals(data_out.aloc[flags_in])


# @settings(max_examples=MAX_EXAMPLES, deadline=None)
# @given(data_field_flagger=dataFieldFlagger(), func_kwargs=flagFuncsKwargs())
# def test_maskingEqualsRemoval(data_field_flagger, func_kwargs):
#     """
#     calling a function on pre-flagged data should yield the same
#     results as calling this function on data where the flagged values
#     are removed
#     """
#     func, kwargs = func_kwargs

#     data, field, flagger = data_field_flagger
#     flagged_in = flagger.isFlagged(flag=flagger.BAD, comparator=">=")

#     # mask and call
#     data_left, _ = _maskData(data, flagger, columns=[field], to_mask=flagger.BAD)
#     data_left, _ = func(data_left, field, flagger, **kwargs)

#     # remove and call
#     data_right = data.aloc[~flagged_in]
#     flagger_right = flagger.initFlags(flagger.getFlags().aloc[~flagged_in])
#     data_right, _ = func(data_right, field, flagger_right, **kwargs)

#     # NOTE: we need to handle the implicit type conversion in `_maskData`
#     data_left_compare = data_left.aloc[~flagged_in]
#     data_left_compare[field] = data_left_compare[field].astype(data[field].dtype)

#     assert data_right.equals(data_left_compare)
