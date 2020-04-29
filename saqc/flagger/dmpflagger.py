#! /usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import json
from copy import deepcopy
from typing import TypeVar

import dios.dios as dios

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

    def initFlags(self, data: dios.DictOfSeries = None, flags: dios.DictOfSeries = None):
        """
        initialize a flagger based on the given 'data' or 'flags'
        if 'data' is not None: return a flagger with flagger.UNFALGGED values
        if 'flags' is not None: return a flagger with the given flags
        """

        # implicit set self._flags, and make deepcopy of self aka. DmpFlagger
        newflagger = super().initFlags(data=data, flags=flags)
        newflagger._causes = newflagger.flags.astype(str)
        newflagger._comments = newflagger.flags.astype(str)
        newflagger._causes[:], newflagger._comments[:] = "", ""
        return newflagger

    def getFlagger(self, field=None, loc=None):
        newflagger = super().getFlagger(field=field, loc=loc)
        flags = newflagger.flags
        newflagger._causes = self._causes.aloc[flags, ...]
        newflagger._comments = self._comments.aloc[flags, ...]
        return newflagger

    def setFlagger(self, other: DmpFlaggerT, join: str="outer"):
        assert isinstance(other, DmpFlagger)
        out = super().setFlagger(other, join)
        out._causes = mergeDios(out._causes, other._causes, join=join)
        out._comments = mergeDios(out._comments, other._comments, join=join)
        return out

    def setFlags(self, field, loc=None, flag=None, force=False, comment="", cause="", **kwargs):
        assert "iloc" not in kwargs, "deprecated keyword, iloc"
        assertScalar("field", field, optional=False)

        flag = self.BAD if flag is None else flag
        comment = json.dumps(dict(comment=comment, commit=self.project_version, test=kwargs.get("func_name", "")))

        if force:
            row_indexer = loc
        else:
            # trim flags to loc, we always get a pd.Series returned
            this = self.getFlags(field=field, loc=loc)
            row_indexer = this < flag

        out = deepcopy(self)
        out._flags.aloc[row_indexer, field] = flag
        out._causes.aloc[row_indexer, field] = cause
        out._comments.aloc[row_indexer, field] = comment
        return out

