#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from hypothesis import given, settings

from saqc.constants import UNFLAGGED, BAD
from saqc.core.register import FunctionWrapper

from tests.fuzzy.lib import dataFieldFlags, MAX_EXAMPLES


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flags=dataFieldFlags())
def test_maskingMasksData(data_field_flags):
    """
    test if flagged values are replaced by np.nan
    """
    data_in, field, flags = data_field_flags
    data_masked, mask = FunctionWrapper._maskData(
        data_in, flags, columns=[field], thresh=UNFLAGGED
    )  # thresh UNFLAGGED | np.inf
    assert data_masked[field].iloc[mask[field].index].isna().all()
    assert (flags[field].iloc[mask[field].index] > UNFLAGGED).all()


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flags=dataFieldFlags())
def test_dataMutationPreventsUnmasking(data_field_flags):
    """test if (un)masking works as expected on data-changes.

    if `data` is mutated after `_maskData`, `_unmaskData` should be a no-op
    """
    filler = -9999

    data_in, field, flags = data_field_flags

    data_masked, mask = FunctionWrapper._maskData(
        data_in, flags, columns=[field], thresh=UNFLAGGED
    )
    data_masked[field] = filler
    data_out = FunctionWrapper._unmaskData(data_masked, mask)
    assert (data_out[field] == filler).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flags=dataFieldFlags())
def test_flagsMutationPreventsUnmasking(data_field_flags):
    """test if (un)masking works as expected on flags-changes.

    if `flags` is mutated after `_maskData`, `_unmaskData` should be a no-op
    """
    data_in, field, flags = data_field_flags

    data_masked, mask = FunctionWrapper._maskData(
        data_in, flags, columns=[field], thresh=UNFLAGGED
    )
    flags[:, field] = UNFLAGGED
    data_out = FunctionWrapper._unmaskData(data_masked, mask)
    assert (data_out.loc[flags[field] == BAD, field].isna()).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flags=dataFieldFlags())
def test_reshapingPreventsUnmasking(data_field_flags):
    """test if (un)masking works as expected on index-changes.

    If the index of data (and flags) change in the func, the unmasking,
    should not reapply original data, instead take the new data (and flags) as is.
    """

    filler = -1111

    data_in, field, flags = data_field_flags

    data_masked, mask = FunctionWrapper._maskData(
        data_in, flags, columns=[field], thresh=UNFLAGGED
    )
    # mutate indexes of `data` and `flags`
    index = data_masked[field].index.to_series()
    index.iloc[-len(data_masked[field]) // 2 :] += pd.Timedelta("7.5Min")
    data_masked[field] = pd.Series(data=filler, index=index)

    fflags = flags[field]
    flags.drop(field)
    flags[field] = pd.Series(data=fflags.values, index=index)

    data_out = FunctionWrapper._unmaskData(data_masked, mask)
    assert (data_out[field] == filler).all(axis=None)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(data_field_flags=dataFieldFlags())
def test_unmaskingInvertsMasking(data_field_flags):
    """
    unmasking data should invert the masking
    """
    data_in, field, flags = data_field_flags

    data_masked, mask = FunctionWrapper._maskData(
        data_in, flags, columns=[field], thresh=UNFLAGGED
    )
    data_out = FunctionWrapper._unmaskData(data_masked, mask)
    assert pd.DataFrame.equals(
        data_out.to_df().astype(float), data_in.to_df().astype(float)
    )


# @settings(max_examples=MAX_EXAMPLES, deadline=None)
# @given(data_field_flags=dataFieldFlags(), func_kwargs=flagFuncsKwargs())
# def test_maskingPreservesData(data_field_flags, func_kwargs):
#     """
#     no mutations on pre-flagged data

#     calling a function on pre-flagged data should yield the same
#     behavior as calling this function on data where the flagged values
#     are removed
#     """

#     data_in, field, flags = data_field_flags

#     data_masked, mask = _maskData(data_in, flags, columns=[field], dfilter=flags.BAD)
#     func, kwargs = func_kwargs
#     data_masked, _ = func(data_masked, field, flags, **kwargs)
#     data_out = _unmaskData(data_in, mask, data_masked, flags, dfilter=flags.BAD)

#     flags_in = flags.isFlagged(flag=flags.BAD)
#     assert data_in.aloc[flags_in].equals(data_out.aloc[flags_in])


# @settings(max_examples=MAX_EXAMPLES, deadline=None)
# @given(data_field_flags=dataFieldFlags(), func_kwargs=flagFuncsKwargs())
# def test_maskingEqualsRemoval(data_field_flags, func_kwargs):
#     """
#     calling a function on pre-flagged data should yield the same
#     results as calling this function on data where the flagged values
#     are removed
#     """
#     func, kwargs = func_kwargs

#     data, field, flags = data_field_flags
#     flagged_in = flags.isFlagged(flag=flags.BAD, comparator=">=")

#     # mask and call
#     data_left, _ = _maskData(data, flags, columns=[field], dfilter=flags.BAD)
#     data_left, _ = func(data_left, field, flags, **kwargs)

#     # remove and call
#     data_right = data.aloc[~flagged_in]
#     flags_right = flags.initFlags(flags.getFlags().aloc[~flagged_in])
#     data_right, _ = func(data_right, field, flags_right, **kwargs)

#     # NOTE: we need to handle the implicit type conversion in `_maskData`
#     data_left_compare = data_left.aloc[~flagged_in]
#     data_left_compare[field] = data_left_compare[field].astype(data[field].dtype)

#     assert data_right.equals(data_left_compare)
