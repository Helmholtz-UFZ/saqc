#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import io
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TextIO, Tuple
from urllib.request import urlopen

import pandas as pd

from saqc import SaQC
from saqc.exceptions import ParsingError
from saqc.lib.tools import isQuoted
from saqc.parsing.visitor import ConfigFunctionParser


def _readLines(
    it: Iterable[str], column_sep=";", comment_prefix="#", skip=0
) -> pd.DataFrame:
    out = []
    for i, line in enumerate(it):
        if (skip := skip - 1) > 0:
            continue
        if not (row := line.strip().split(comment_prefix, 1)[0]):
            continue
        parts = [p.strip() for p in row.split(column_sep)]
        if len(parts) != 2:
            raise ParsingError(
                f"The configuration format expects exactly two "
                f"columns, one for the variable name and one for "
                f"the tests, but {len(parts)} columns were found "
                f"in line {i}.\n\t{line!r}"
            )
        out.append([i + 1] + parts)
    if not out:
        raise ParsingError("Config file is empty")
    return pd.DataFrame(out[1:], columns=["lineno", "varname", "test"]).set_index(
        "lineno"
    )


def readFile(fname, skip=1) -> pd.DataFrame:
    """Read and parse a config file to a DataFrame"""

    def _open(file_or_buf) -> TextIO:
        if not isinstance(file_or_buf, (str, Path)):
            return file_or_buf
        try:
            fh = io.open(file_or_buf, "r", encoding="utf-8")
        except (OSError, ValueError):
            fh = io.StringIO(urlopen(str(file_or_buf)).read().decode("utf-8"))
            fh.seek(0)
        return fh

    def _close(fh):
        try:
            fh.close()
        except AttributeError:
            pass

    # mimic `with open(): ...`
    file = _open(fname)
    try:
        return _readLines(file, skip=skip)
    finally:
        _close(file)


def fromConfig(fname, *args, **func_kwargs):
    return _ConfigReader(*args, **func_kwargs).readCsv(fname).run()


class _ConfigReader:
    logger: logging.Logger
    saqc: SaQC
    file: str | None
    config: pd.DataFrame | None
    parsed: List[Tuple[Any, ...]] | None
    regex: bool | None
    varname: str | None
    lineno: int | None
    field: str | None
    test: str | None
    func: str | None
    func_kws: Dict[str, Any] | None

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.saqc = SaQC(*args, **kwargs)
        self.file = None
        self.config = None
        self.parsed = None
        self.regex = None
        self.varname = None
        self.lineno = None
        self.field = None
        self.test = None
        self.func = None
        self.func_kws = None

    def readCsv(self, file: str, skip=1):
        self.logger.debug(f"opening csv file: {file}")
        self.config = readFile(file, skip=skip)
        self.file = file
        return self

    def readRecords(self, seq: Sequence[Dict[str, Any]]):
        self.logger.debug(f"read records: {seq}")
        df = pd.DataFrame.from_records(seq)
        df.columns = ["varname", "func", "kwargs"]
        kws = df["kwargs"].apply(
            lambda e: ", ".join([f"{k}={v}" for k, v in e.items()])
        )
        df["test"] = df["func"] + "(" + kws + ")"
        self.config = df.loc[:, ["varname", "test"]].copy()
        return self

    def _readJson(self, d, unpack):
        if unpack is not None:
            d = unpack(d)
        elif isinstance(d, dict):
            raise TypeError("parsed json resulted in a dict, but a array/list is need")
        return self.readRecords(d)

    def readJson(self, file: str, unpack: callable | None = None):
        self.logger.debug(f"opening json file: {file}")
        with open(file, "r") as fh:
            d = json.load(fh)
        self.file = file
        return self._readJson(d, unpack)

    def readJsonString(self, jn: str, unpack: callable | None = None):
        self.logger.debug(f"read json string: {jn}")
        d = json.loads(jn)
        return self._readJson(d, unpack)

    def readString(self, s: str, line_sep="\n", column_sep=";"):
        self.logger.debug(f"read config string: {s}")
        lines = s.split(line_sep)
        self.config = _readLines(lines, column_sep=column_sep)
        return self

    def _parseLine(self):
        self.logger.debug(f"parse line {self.lineno}: {self.varname!r}; {self.test!r}")
        self.regex = isQuoted(self.varname)
        self.field = self.varname[1:-1] if self.regex else self.varname

        try:
            tree = ast.parse(self.test, mode="eval").body
            func, kws = ConfigFunctionParser().parse(tree)
        except Exception as e:
            # We raise a NEW exception here, because the
            # traceback hold no relevant info for a CLI user.
            err = type(e) if isinstance(e, NameError) else ParsingError
            meta = self._getFormattedInfo(
                "The exception occurred during parsing of a config"
            )
            if hasattr(e, "add_note"):  # python 3.11+
                e = err(*e.args)
                e.add_note(meta)
                raise e from None
            raise err(str(e) + meta) from None

        if "field" in kws:
            kws["target"] = self.field
        else:
            kws["field"] = self.field
        self.func = func
        self.func_kws = kws

    def _execLine(self):
        self.logger.debug(
            f"execute line {self.lineno}: {self.varname!r}; {self.test!r}"
        )
        # We explicitly route all function calls through SaQC.__getattr__
        # in order to do a FUNC_MAP lookup. Otherwise, we wouldn't be able
        # to overwrite existing test functions with custom register calls.
        try:
            self.saqc = self.saqc.__getattr__(self.func)(
                regex=self.regex, **self.func_kws
            )
        except Exception as e:
            # We use a special technique for raising here, because
            # we want this location of rising, line up in the traceback,
            # instead of showing up at last (most relevant). Also, we
            # want to include some meta information about the config.
            meta = self._getFormattedInfo(
                "The exception occurred during execution of a config"
            )
            if hasattr(e, "add_note"):  # python 3.11+
                e.add_note(meta)
                raise e
            raise type(e)(str(e) + meta).with_traceback(e.__traceback__) from None

    def _getFormattedInfo(self, msg=None, indent=2):
        prefix = " " * indent
        info = textwrap.indent(
            f"file:    {self.file!r}\n"
            f"line:    {self.lineno}\n"
            f"varname: {self.varname!r}\n"
            f"test:    {self.test!r}\n",
            prefix,
        )
        if msg:
            info = textwrap.indent(f"{msg}\n{info}", prefix)
        return f"\n{info}"

    def run(self):
        """Parse and execute a config line by line."""
        assert self.config is not None
        for lineno, varname, test in self.config.itertuples():
            self.lineno = lineno
            self.varname = varname
            self.test = test
            self._parseLine()
            self._execLine()
        return self.saqc
