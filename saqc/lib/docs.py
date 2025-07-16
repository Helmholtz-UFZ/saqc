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
        "description": "Variable to process.",
        "typehint": "",
    },
    "target": {
        "name": "target",
        "description": "Variable name to which the results are written. :py:attr:`target` will be created if it does not exist. Defaults to :py:attr:`field`.",
        "typehint": ":py:class:`SaQCFields` | :py:class:`newSaQCFields` ",
        "optional": True,
    },
    "dfilter": {
        "name": "dfilter",
        "description": "Defines which observations will be masked based on the already existing flags. Any data point with a flag equal or worse to this threshold will be passed as ``NaN`` to the function. Defaults to the ``DFILTER_ALL`` value of the translation scheme.",
        "typehint": ":py:class:`Any`",
        "optional": True,
    },
    "flag": {
        "name": "flag",
        "description": "The flag value the function uses to mark observations. Defaults to the ``BAD`` value of the translation scheme.",
        "typehint": ":py:class:`Any`",
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
    for p in ("target", "dfilter", "flag"):
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
