#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from copy import deepcopy
from typing import TypeVar, Optional, List

import pandas as pd

from dios import DictOfSeries

from saqc.flagger.categoricalflagger import CategoricalFlagger
from saqc.lib.tools import assertScalar, mergeDios

DmpFlaggerT = TypeVar("DmpFlaggerT")


class Keywords:
    VERSION = "$version"


class FlagFields:
    FLAG = "quality_flag"
    CAUSE = "quality_cause"
    COMMENT = "quality_comment"


class ColumnLevels:
    VARIABLES = "variables"
    FLAGS = "flags"


FLAGS = ["NIL", "OK", "DOUBTFUL", "BAD"]


class DmpFlagger(CategoricalFlagger):
    def __init__(self, **kwargs):
        super().__init__(FLAGS)
        self.flags_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]
        self.extra_defaults = dict(cause=FLAGS[0], comment="")
        self.signature = ("flag", "comment", "cause", "force")

        self._global_comments = kwargs
        self._flags = None
        self._causes = None
        self._comments = None

    @property
    def causes(self):
        return self._causes

    @property
    def comments(self):
        return self._comments

    def toFrame(self):
        out = pd.concat(
            [self._flags.to_df(), self._causes.to_df(), self._comments.to_df()],
            axis=1,
            keys=[FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT],
        )
        out = out.reorder_levels(order=[1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
        return out

    def initFlags(self, data: DictOfSeries = None, flags: DictOfSeries = None):
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        # implicit set self._flags, and make deepcopy of self aka. DmpFlagger
        newflagger = super().initFlags(data=data, flags=flags)
        newflagger._causes = newflagger._flags.astype(str)
        newflagger._comments = newflagger._flags.astype(str)
        newflagger._causes[:], newflagger._comments[:] = "", ""
        return newflagger

    def slice(self, field=None, loc=None, drop=None, inplace=False):
        newflagger = super().slice(field=field, loc=loc, drop=drop, inplace=inplace)
        flags = newflagger._flags
        newflagger._causes = self._causes.aloc[flags, ...]
        newflagger._comments = self._comments.aloc[flags, ...]
        return newflagger

    def merge(self, other: DmpFlagger, subset: Optional[List] = None, join: str = "merge", inplace=False):
        assert isinstance(other, DmpFlagger)
        flags = mergeDios(self._flags, other._flags, subset=subset, join=join)
        causes = mergeDios(self._causes, other._causes, subset=subset, join=join)
        comments = mergeDios(self._comments, other._comments, subset=subset, join=join)
        if inplace:
            self._flags = flags
            self._causes = causes
            self._comments = comments
            return self
        else:
            return self._construct_new(flags, causes, comments)

    def getFlags(self, field=None, loc=None, full=False):
        # loc should be a valid 2D-indexer and
        # then field must be None. Otherwise aloc
        # will fail and throw the correct Error.
        if isinstance(loc, DictOfSeries) and field is None:
            indexer = loc
        else:
            loc = slice(None) if loc is None else loc
            field = slice(None) if field is None else self._check_field(field)
            indexer = (loc, field)

        # this is a bug in `dios.aloc`, which may return a shallow copied dios, if `slice(None)` is passed
        # as row indexer. Thus is because pandas `.loc` return a shallow copy if a null-slice is passed to a series.
        flags = self._flags.aloc[indexer].copy()

        if full:
            causes = self._causes.aloc[indexer].copy()
            comments = self._comments.aloc[indexer].copy()
            return flags, dict(cause=causes, comment=comments)
        else:
            return flags

    def setFlags(
        self,
        field,
        loc=None,
        flag=None,
        cause="OTHER",
        comment="",
        force=False,
        inplace=False,
        with_extra=False,
        flag_after=None,
        flag_before=None,
        win_flag=None,
        **kwargs
    ):
        assert "iloc" not in kwargs, "deprecated keyword, iloc"
        assertScalar("field", self._check_field(field), optional=False)

        out = self if inplace else deepcopy(self)

        if with_extra:
            for val in [comment, cause, flag]:
                if not isinstance(val, pd.Series):
                    raise TypeError(f"`flag`, `cause`, `comment` must be pd.Series, if `with_extra=True`.")
            assert flag.index.equals(comment.index) and flag.index.equals(cause.index)

        else:
            flag = self.BAD if flag is None else flag
            comment = json.dumps(
                {**self._global_comments,
                 "comment": comment,
                 "test": kwargs.get("func_name", "")}
            )

        flags = self.getFlags(field=field, loc=loc)
        if force:
            mask = pd.Series(True, index=flags.index, dtype=bool)
        else:
            mask = flags < flag

        # set flags of the test
        out._flags.aloc[mask, field] = flag
        out._causes.aloc[mask, field] = cause
        out._comments.aloc[mask, field] = comment

        # calc and set window flags
        if flag_after is not None or flag_before is not None:
            win_mask, win_flag = self._getWindowMask(field, mask, flag_after, flag_before, win_flag, flag, force)
            out._flags.aloc[win_mask, field] = win_flag
            out._causes.aloc[win_mask, field] = cause
            out._comments.aloc[win_mask, field] = comment

        return out

    def replaceField(self, field, flags, inplace=False, cause=None, comment=None, **kwargs):
        """ Replace or delete all data for a given field.

        Parameters
        ----------
        field : str
            The field to replace / delete. If the field already exist, the respected data
            is replaced, otherwise the data is inserted in the respected field column.
        flags : pandas.Series or None
            If None, the series denoted by `field` will be deleted. Otherwise
            a series of flags (dtype flagger.dtype) that will replace the series
            currently stored under `field`
        causes : pandas.Series
            A series of causes (dtype str).
        comments : pandas.Series
            A series of comments (dtype str).
        inplace : bool, default False
            If False, a flagger copy is returned, otherwise the flagger is not copied.
        **kwargs : dict
            ignored.

        Returns
        -------
        flagger: saqc.flagger.BaseFlagger
            The flagger object or a copy of it (if inplace=True).

        Raises
        ------
        ValueError: (delete) if field does not exist
        TypeError: (replace / insert) if flags, causes, comments are not pd.Series
        AssertionError: (replace / insert) if flags, causes, comments does not have the same index

        Notes
        -----
        If deletion is requested (`flags=None`), `causes` and `comments` are don't-care.

        Flags, causes and comments must have the same index, if flags is not None, also
        each is casted implicit to the respected dtype.
        """
        assertScalar("field", field, optional=False)
        out = self if inplace else deepcopy(self)
        causes, comments = cause, comment

        # delete
        if flags is None:
            if field not in self._flags:
                raise ValueError(f"{field}: field does not exist")
            del out._flags[field]
            del out._comments[field]
            del out._causes[field]

        # insert / replace
        else:
            for val in [flags, causes, comments]:
                if not isinstance(val, pd.Series):
                    raise TypeError(f"`flag`, `cause`, `comment` must be pd.Series.")
            assert flags.index.equals(comments.index) and flags.index.equals(causes.index)
            out._flags[field] = flags.astype(self.dtype)
            out._causes[field] = causes.astype(str)
            out._comments[field] = comments.astype(str)
        return out

    def _construct_new(self, flags, causes, comments) -> DmpFlagger:
        new = DmpFlagger()
        new._global_comments = self._global_comments
        new._flags = flags
        new._causes = causes
        new._comments = comments
        return new

    @property
    def SUSPICIOUS(self):
        return FLAGS[-2]
