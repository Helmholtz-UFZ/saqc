# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re

FUNC_NAPOLEAN_STYLE_ORDER = [
    "Head",
    "Parameters",
    "Returns",
    "Notes",
    "See also",
    "Examples",
    "References",
]


def doc(doc_string: str, template="saqc_methods", source="function_string"):
    def docFunc(meth):
        if template == "saqc_methods":
            meth.__doc__ = saqcMethodsTemplate(doc_string, source)
        return meth

    return docFunc


def getDocstringIndent(lines: list) -> str:
    """returns a whitespace string matching the indent size of the passed docstring_list"""
    for line in lines:
        if len(line) == 0 or re.match(" *$", line):
            continue
        return re.match(" *", line)[0]
    return ""


def getSections(doc_string: list, indent_str: str) -> dict:
    """Returns a dictionary of sections, with section names as keys"""
    section_lines = [0]
    section_headings = ["Head"]
    for k in range(len(doc_string) - 1):
        # check if next line is an underscore line (section signator):
        if re.match(indent_str + "-+$", doc_string[k + 1]):
            # check if underscore length matches heading length
            if len(doc_string[k + 1]) == len(doc_string[k]):
                section_lines.append(k)
                # skip leading whitespaces
                skip = re.match("^ *", doc_string[k]).span()[-1]
                section_headings.append(doc_string[k][skip:])
    section_lines.append(len(doc_string))
    section_content = [
        doc_string[section_lines[k] : section_lines[k + 1]]
        for k in range(len(section_lines) - 1)
    ]
    section_content = [clearTrailingWhitespace(p) for p in section_content]
    sections = dict(zip(section_headings, section_content))
    return sections


def getParameters(section: list, indent_str: str) -> dict:
    """Returns a dictionary of Parameter documentations, with parameter names as keys"""
    parameter_lines = []
    parameter_names = []
    for k in range(len(section)):
        # try catch a parameter definition start (implicitly assuming parameter names have no
        # whitespaces):
        param = re.match(indent_str + r"(\S+) *:", section[k])
        if param:
            parameter_lines.append(k)
            parameter_names.append(param.group(1))

    parameter_lines.append(len(section))
    parameter_content = [
        section[parameter_lines[k] : parameter_lines[k + 1]]
        for k in range(len(parameter_lines) - 1)
    ]
    parameter_content = [clearTrailingWhitespace(p) for p in parameter_content]
    parameter_dict = dict(zip(parameter_names, parameter_content))
    return parameter_dict


def mkParameter(
    parameter_name: str, parameter_type: str, parameter_doc: str, indent_str: str
) -> dict:
    parameter_doc = parameter_doc.splitlines()
    parameter_doc = [indent_str + " " * 4 + p for p in parameter_doc]
    content = [indent_str + f"{parameter_name} : {parameter_type}"]
    content += parameter_doc
    return {parameter_name: content}


def makeSection(section_name: str, indent_str: str, doc_content: str = None) -> dict:
    content = [indent_str + section_name]
    content += [indent_str + "_" * len(section_name)]
    content += [" "]
    if doc_content:
        content += doc_content.splitlines()

    return {section_name: content}


def composeDocstring(
    section_dict: dict, order: list = FUNC_NAPOLEAN_STYLE_ORDER
) -> str:
    """Compose final docstring from a sections dictionary"""
    doc_string = []
    section_dict = section_dict.copy()
    for sec in order:
        dc = section_dict.pop(sec, [])
        doc_string += dc
        # blank line at section end
        if len(dc) > 0:
            doc_string += [""]

    return "\n".join(doc_string)


def clearTrailingWhitespace(doc: list) -> list:
    """Clears trailing whitespace lines"""
    for k in range(len(doc), 0, -1):
        if not re.match(r"^\s*$", doc[k - 1]):
            break
    return doc[:k]


def rmParameter(para_name: str, section: list, indent_str: str):
    params = getParameters(section, indent_str=indent_str)
    if para_name in params:
        rm_idx = [
            sidx
            for sidx in range(len(section))
            if params[para_name] == section[sidx : sidx + len(params[para_name])]
        ][0]
        section = section[:rm_idx] + section[rm_idx + len(params[para_name]) :]
    return section


def saqcMethodsTemplate(doc_string: str, source="function_string"):
    if source == "function_string":
        doc_string = doc_string.splitlines()
        indent_string = getDocstringIndent(doc_string)
        sections = getSections(doc_string, indent_str=indent_string)
        # modify returns section
        sections.pop("Returns", None)
        returns_section = makeSection(section_name="Returns", indent_str=indent_string)
        out_para = mkParameter(
            parameter_name="out",
            parameter_type="saqc.SaQC",
            parameter_doc="An :py:meth:`saqc.SaQC` object, holding the (possibly) modified data",
            indent_str=indent_string,
        )
        returns_section["Returns"] += out_para["out"]
        sections.update(returns_section)
        # remove flags and data parameter from docstring
        if "Parameters" in sections:
            para_sec = sections["Parameters"]
            sections.pop("Parameters", None)
            para_sec = rmParameter("data", para_sec, indent_string)
            para_sec = rmParameter("flags", para_sec, indent_string)
            sections["Parameters"] = para_sec

        doc_string = composeDocstring(
            section_dict=sections, order=FUNC_NAPOLEAN_STYLE_ORDER
        )
    return doc_string
