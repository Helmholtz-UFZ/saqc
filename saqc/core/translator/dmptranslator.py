#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from dios.dios.dios import DictOfSeries

import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from saqc.core.lib import SaQCFunction, ColumnSelector
from saqc.core.flags import (
    Flags,
    UNFLAGGED,
    GOOD,
    DOUBTFUL,
    BAD,
)
from saqc.lib.types import MaterializedGraph
from saqc.core.translator.basetranslator import Translator, ForwardMap


class DmpTranslator(Translator):

    """
    Implements the translation from and to the flagging scheme implemented in
    the UFZ - Datamanagementportal
    """

    ARGUMENTS = {"comment": "", "cause": ""}

    _FORWARD: ForwardMap = {
        "NIL": UNFLAGGED,
        "OK": GOOD,
        "DOUBTFUL": DOUBTFUL,
        "BAD": BAD,
    }

    _QUALITY_CAUSES = {
        "BATTERY_LOW",
        "BELOW_MINIMUM",
        "ABOVE_MAXIMUM",
        "BELOW_OR_ABOVE_MIN_MAX",
        "ISOLATED_SPIKE",
        "DEFECTIVE_SENSOR",
        "LEFT_CENSORED_DATA",
        "RIGHT_CENSORED_DATA",
        "OTHER",
        "AUTO_FLAGGED",
    }

    def __init__(self):
        raise NotImplementedError
        super().__init__(
            forward=self._FORWARD, backward={v: k for k, v in self._FORWARD.items()}
        )

    @staticmethod
    def _getFieldFunctions(
        field: str, call_stack: MaterializedGraph
    ) -> List[SaQCFunction]:
        """
        Return the names of all functions called on `field`

        Parameters
        ----------
        field: str
            variable/column name

        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)

        Note
        ----
        Could (and maybe should) be implemented as a method of `CallGraph`

        Currently we work around the issue, that we keep track of the
        computations we do on a variable using the variable name, but also
        allow mutations of that name (i.e. our key) through `tools.rename`
        in a somewhat hacky way. There are better ideas, to solve this (i.e.
        global function pointers), but for the moment this has to do the trick
        """
        # backtrack name changes and let's look, if our field
        # originally had another name
        for sel, func in call_stack[::-1]:
            if func.name == "tools.rename":
                new_name = func.keywords.get("new_name") or func.args[3]
                if new_name == field:
                    field = sel.field

        out = [SaQCFunction(name="")]
        for sel, func in call_stack:
            if sel.field == field:
                out.append(func)
                # forward track name changes
                if func.name == "tools.rename":
                    field = func.keywords.get("new_name") or func.args[3]

        return out

    def forward(self, flags: pd.DataFrame) -> Tuple[Flags, MaterializedGraph]:
        """
        Translate from 'extrnal flags' to 'internal flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate

        Returns
        -------
        Flags object
        """
        cols = flags.columns
        if not isinstance(cols, pd.MultiIndex):
            raise TypeError("DMP-Flags need mult-index columns")
        col_labels = {"quality_flag", "quality_comment", "quality_cause"}
        if set(cols.get_level_values(1)) != col_labels:
            raise TypeError(
                f"DMP-Flags expect the labels '{list(col_labels)}' in the secondary level"
            )

        qflags = flags.xs(key="quality_flag", axis="columns", level=1)

        # We want to build a call graph from the given flags and as the DMP flags
        # contain the name of last function that set a certain flag, we want to
        # leverage this information
        graph: MaterializedGraph = []

        for field in qflags.columns:

            # extract relevant information from the comments
            data = pd.DataFrame(
                flags.loc[:, (field, "quality_comment")].apply(json.loads).to_list(),
                index=flags.index,
            )
            data["causes"] = flags.loc[:, (field, "quality_cause")]

            loc = ColumnSelector(field=field, target="field", regex=False)

            # we can't infer information about the ordering of function calls,
            # so we order the history by appearance
            # for _, group in data.fillna("").groupby(["test", "comment", "causes"]):
            for _, group in data.loc[data["test"].replace("", np.nan).notna()].groupby(
                ["test", "comment", "causes"]
            ):
                fname, comment, cause = group.iloc[0]
                func = SaQCFunction(
                    name=fname,
                    function=Translator._generateInitFunction(
                        field, qflags.loc[group.index]
                    ),
                    comment=comment,
                    cause=cause,
                )
                graph.append((loc, func))

        tflags, _ = super().forward(qflags)
        return tflags, graph

    def backward(self, flags: Flags, call_graph: MaterializedGraph) -> pd.DataFrame:
        """
        Translate from 'internal flags' to 'external flags'

        Parameters
        ----------
        flags : pd.DataFrame
            The external flags to translate
        call_stack : List
            The saqc functions called to generate the given `flags` (i.e. `SaQC._computed`)

        Returns
        -------
        pd.DataFrame
        """
        tflags = super().backward(flags, call_graph)

        out = {}
        for field in tflags.columns:
            flag_call_history = self._getFieldFunctions(field, call_graph)
            flag_pos = flags.history[field].idxmax()
            comments, causes = [], []
            # NOTE:
            # Strangely enough, this loop withstood all my efforts
            # to speed it up through vectorization - the simple
            # loop always outperformed even careful `pd.DataFrame.apply`
            # versions. The latest try is left as a comment below.
            for p in flag_pos:
                func = flag_call_history[p]
                cause = func.keywords.get("cause", self.ARGUMENTS["cause"])
                comment = json.dumps(
                    {
                        "test": func.name,
                        "comment": func.keywords.get(
                            "comment", self.ARGUMENTS["comment"]
                        ),
                    }
                )
                causes.append(cause)
                comments.append(comment)

            # DMP quality_cause needs some special care as only certain values
            # and combinations are allowed.
            # See: https://wiki.intranet.ufz.de/wiki/dmp/index.php/Qualit%C3%A4tsflags
            causes = pd.Series(causes, index=flags[field].index)
            causes[
                (causes == self.ARGUMENTS["cause"]) & (flags[field] > GOOD)
            ] = "OTHER"
            if not ((causes == "") | causes.isin(self._QUALITY_CAUSES)).all():
                raise ValueError(
                    f"quality causes needs to be one of {self._QUALITY_CAUSES}"
                )

            var_flags = {
                "quality_flag": tflags[field],
                "quality_comment": pd.Series(comments, index=flags[field].index),
                "quality_cause": causes,
            }
            out[field] = pd.DataFrame(var_flags)
        return pd.concat(out, axis="columns")

        # for field in tflags.columns:
        #     call_history = []
        #     for func in self._getFieldFunctions(field, call_graph):
        #         func_info = {
        #             "cause": func.keywords.get("cause", self.ARGUMENTS["comment"]),
        #             "comment": json.dumps({
        #                 "test": func.name,
        #                 "comment": func.keywords.get("comment", self.ARGUMENTS["comment"]),
        #             })
        #          }
        #         call_history.append(func_info)

        #     functions = pd.DataFrame(call_history)
        #     flag_pos = flags.history[field].idxmax()

        #     var_flags = {
        #         "quality_flag": tflags[field].reset_index(drop=True),
        #         "quality_comment": functions.loc[flag_pos, "comment"].reset_index(drop=True),
        #         "quality_cause": functions.loc[flag_pos, "cause"].reset_index(drop=True),
        #     }
        #     out[field] = pd.DataFrame(var_flags, index=flag_pos.index)
        # return pd.concat(out, axis="columns")
