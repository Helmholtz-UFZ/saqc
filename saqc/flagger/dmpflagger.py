#! /usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import json
from copy import deepcopy
from typing import TypeVar, Optional, List

import pandas as pd

import dios

from saqc.flagger.baseflagger import diosT
from saqc.flagger.categoricalflagger import CategoricalFlagger
from saqc.lib.tools import assertScalar, mergeDios, mutateIndex

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
    def __init__(self):
        super().__init__(FLAGS)
        self.flags_fields = [FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT]
        version = subprocess.run(
            "git describe --tags --always --dirty", shell=True, check=False, stdout=subprocess.PIPE,
        ).stdout
        self.project_version = version.decode().strip()
        self.signature = ("flag", "comment", "cause", "force")
        self._flags = None
        self._causes = None
        self._comments = None

    @property
    def causes(self):
        return self._causes

    @property
    def comments(self):
        return self._comments

    def getFlagsAll(self):
        out = pd.concat(
            [self._flags.to_df(), self._causes.to_df(), self._comments.to_df()],
            axis=1,
            keys=[FlagFields.FLAG, FlagFields.CAUSE, FlagFields.COMMENT],
        )
        out = out.reorder_levels(order=[1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
        return out

    def initFlags(self, data: dios.DictOfSeries = None, flags: dios.DictOfSeries = None):
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

    def rename(self, field: str, new_name: str, inplace=False):
        newflagger = super().rename(field, new_name, inplace=inplace)
        newflagger._causes.columns = newflagger._flags.columns
        newflagger._comments.columns = newflagger._flags.columns
        return newflagger

    def merge(self, other: DmpFlaggerT, subset: Optional[List] = None, join: str = "merge", inplace=False):
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
        if isinstance(loc, diosT) and field is None:
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
                {"comment": comment,
                 "commit": self.project_version,
                 "test": kwargs.get("func_name", "")}
            )

        if force:
            row_indexer = slice(None) if loc is None else loc
        else:
            this = self.getFlags(field=field, loc=loc)
            row_indexer = this < flag

        out._flags.aloc[row_indexer, field] = flag
        out._causes.aloc[row_indexer, field] = cause
        out._comments.aloc[row_indexer, field] = comment
        return out

    def _construct_new(self, flags, causes, comments) -> DmpFlaggerT:
        new = DmpFlagger()
        new.project_version = self.project_version
        new._flags = flags
        new._causes = causes
        new._comments = comments
        return new
