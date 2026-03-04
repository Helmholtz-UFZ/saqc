# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import TypedDict

from docstring_parser import (
    DocstringParam,
    DocstringReturns,
    DocstringStyle,
    compose,
    parse,
)


class ParamDict(TypedDict):
    typehint: str | None
    description: str | None
    optional: bool | None


DOC_TEMPLATES = {
    "field": {
        "typehint": "",
        "description": "List of variables names to process.",
    },
    "target": {"optional": False},
}

COMMON = {
    "field": {
        "name": "field",
        "description": "Name of the input variable to process.",
        "typehint": "",
    },
    "target": {
        "name": "target",
        "description": "Name of the variable to which the results are written. If the variable does not exist, it will be created. Defaults to ``field``.",
        "typehint": ":py:class:`SaQCFields` | :py:class:`newSaQCFields` ",
        "optional": True,
    },
    "dfilter": {
        "name": "dfilter",
        "description": "Defines which observations are masked based on their existing flags. Any data point with a flag value greater than or equal to this threshold is passed to the function as ``numpy.nan``. Defaults to the ``DFILTER_DEFAULT`` value of the active flagging scheme.",
        "typehint": ":py:class:`Any`",
        "optional": True,
    },
    "flag": {
        "name": "flag",
        "description": "Flag value used to annotate detected observations. Defaults to the ``BAD`` value of the active flagging scheme.",
        "typehint": ":py:class:`Any`",
        "optional": True,
    },
    "start_date": {
        "name": "start_date",
        "description": "Lower temporal bound for function execution. Only observations with timestamps greater than or equal to ``start_date`` are processed. String inputs may be partially specified (e.g., ``'15:00'``, ``'01T12:00'``, ``'01-01'``) to restrict recurring temporal patterns.",
        "typehint": "pd.Timestamp | datetime.datetime | str",
        "optional": True,
    },
    "end_date": {
        "name": "end_date",
        "description": "Upper temporal bound for function execution. Only observations with timestamps less than or equal to ``end_date`` are processed. String inputs may be partially specified to restrict recurring temporal patterns.",
        "typehint": ":py:class:`pd.Timestamp` | :py:class:`datetime.datetime` | :py:class:`str`",
        "optional": True,
    },
}


def toParameter(
    name: str, typehint: str, description: str, optional: bool = False
) -> DocstringParam:
    return DocstringParam(
        args=["param", name],
        description=description,
        arg_name=name,
        type_name=typehint,
        is_optional=optional,
        default=None,
    )


def docurator(func, defaults: dict[str, ParamDict] | None = None):
    if defaults is None:
        defaults = {}

    docstring_return = DocstringReturns(
        args=["returns"],
        description="the updated SaQC object",
        type_name="saqc.SaQC",
        is_generator=False,
        return_name="SaQC",
    )

    tree = parse(func.__doc__, style=DocstringStyle.NUMPYDOC)

    if tree.returns:
        raise ValueError(
            f"'{func.__name__}' function doctstring should not provide a returns section"
        )

    # rewrite parameters
    meta = [toParameter(**{**COMMON["field"], **defaults.get("field", {})})]
    for p in tree.params:
        if p.arg_name in COMMON:
            raise ValueError(
                f"'{func.__name__}' function docstring should not provide a parameter description for '{p.arg_name}'"
            )
        meta.append(p)

    # additional parameters
    for p in ("target", "dfilter", "flag", "start_date", "end_date"):
        meta.append(toParameter(**{**COMMON[p], **defaults.get(p, {})}))

    # return sections
    meta.append(docstring_return)

    # everyhing else the docstring provides
    for m in tree.meta:
        if not isinstance(m, DocstringParam):
            meta.append(m)

    tree.meta = meta

    func.__doc__ = compose(tree)

    return func
