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
    def doc_func(meth):
        if template == "saqc_methods":
            meth.__doc__ = saqc_methods_template(doc_string, source)
        return meth

    return doc_func


def get_docstring_indent(doc_string: list) -> str:
    """returns a whitespace string matching the indent size of the passed docstring_list"""
    regular_line = False
    current_line = 0
    while not regular_line:
        # check if line isnt just a newline string and also not a blank line (with all white space)
        if (not len(doc_string[current_line]) == 0) and (
            not re.match(" *$", doc_string[current_line])
        ):
            regular_line = True
        else:
            current_line += 1
    # get indent-string (smth. like "   ")
    indent_str = re.match(" *", doc_string[current_line])[0]
    return indent_str


def get_sections(doc_string: list, indent_str: str) -> dict:
    """Returns a dictionary of sections, with section names as keys"""
    section_lines = [0]
    section_headings = ["Head"]
    for k in range(len(doc_string) - 1):
        # check if next line is an underscore line (section signator):
        if re.match(indent_str + "-+$", doc_string[k + 1]):
            # check if underscore length matches heading length
            if len(doc_string[k + 1]) == len(doc_string[k]):
                section_lines.append(k)
                # skip leading whitespaces (start with whitespace and end with anything but whitespace
                skip = re.match("^ *", doc_string[k]).span()[-1]
                section_headings.append(doc_string[k][skip:])
    section_lines.append(len(doc_string))
    section_content = [
        doc_string[section_lines[k] : section_lines[k + 1]]
        for k in range(len(section_lines) - 1)
    ]
    section_content = [clear_whitespace_tail(p) for p in section_content]
    sections = dict(zip(section_headings, section_content))
    return sections


def get_parameters(section: list, indent_str: str) -> dict:
    """Returns a dictionary of Parameter documentations, with parameter names as keys"""
    parameter_lines = []
    parameter_names = []
    for k in range(len(section)):
        # try catch a parameter definition start (implicitly assuming parameter names have no
        # whitespaces):
        para = re.match(indent_str + "(\S+) *:", section[k])
        # got one ?
        if para:
            parameter_lines.append(k)
            parameter_names.append(para.group(1))

    parameter_lines.append(len(section))
    parameter_content = [
        section[parameter_lines[k] : parameter_lines[k + 1]]
        for k in range(len(parameter_lines) - 1)
    ]
    parameter_content = [clear_whitespace_tail(p) for p in parameter_content]
    parameter_dict = dict(zip(parameter_names, parameter_content))
    return parameter_dict


def mk_parameter(
    parameter_name: str, parameter_type: str, parameter_doc: str, indent_str: str
) -> dict:
    parameter_doc = parameter_doc.splitlines()
    parameter_doc = [indent_str + " " * 4 + p for p in parameter_doc]
    content = [indent_str + f"{parameter_name} : {parameter_type}"]
    content += parameter_doc
    return {parameter_name: content}


def mk_section(section_name: str, indent_str: str, doc_content: str = None) -> dict:
    content = [indent_str + section_name]
    content += [indent_str + "_" * len(section_name)]
    content += [" "]
    if doc_content:
        content += doc_content.splitlines()

    return {section_name: content}


def compose_docstring(
    section_dict: dict, order: list = FUNC_NAPOLEAN_STYLE_ORDER
) -> str:
    """Compose final docstring from a sections dictionary"""
    doc_string = []
    section_dict = section_dict.copy()
    for sec in order:
        # append section
        dc = section_dict.pop(sec, [])
        doc_string += dc
        # blank line at section end
        if len(dc) > 0:
            doc_string += [""]

    return "\n".join(doc_string)


def clear_whitespace_tail(doc: list) -> list:
    """Clears tailing whitespace lines"""
    for k in range(len(doc), 0, -1):
        if not re.match("^\s*$", doc[k - 1]):
            break
    return doc[:k]


def saqc_methods_template(doc_string: str, source="function_string"):
    if source == "function_string":
        # splitline doc string:
        doc_string = doc_string.splitlines()
        # retrieve doc indention
        indent_string = get_docstring_indent(doc_string)
        # retrieve sections dict
        sections = get_sections(doc_string, indent_str=indent_string)
        # remove returns section (if present)
        sections.pop("Returns", None)
        # generate new (blank) returns section
        returns_section = mk_section(section_name="Returns", indent_str=indent_string)
        # generate return parameter 'out'
        out_para = mk_parameter(
            parameter_name="out",
            parameter_type="saqc.SaQC",
            parameter_doc="An :py:meth:`saqc.SaQC` object, holding the (possibly) modified data",
            indent_str=indent_string,
        )
        # add out - parameter description content to new returns section
        returns_section["Returns"] += out_para["out"]
        # add new returns section to sections dict
        sections.update(returns_section)
        # compose output string with reference to desired order
        doc_string = compose_docstring(
            section_dict=sections, order=FUNC_NAPOLEAN_STYLE_ORDER
        )
    return doc_string
