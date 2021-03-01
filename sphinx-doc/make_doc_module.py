import ast
import os
import click
import pkgutil
import shutil
import re
from collections import OrderedDict
import pickle

new_line_re = "(\r\n|[\r\n])"

doc_mod_structure = {'BasicFlagging': ['outliers.flagRange',
                                       'breaks.flagMissing'],
                     'BasicFlagging_dcstring': '',
                     'AdvancedFlagging': ['pattern.flagPatternByDTW',
                                          'outliers.flagOffset'],
                     'AdvancedFlagging_dcstring': ''}


def rm_section(dcstring, section, _return_section=False):
    """
    Detects a section in a docstring and (default) removes it, or (_return_section=True) returns it
    """
    section_re = (f'{new_line_re}(?P<s_name>[^\n\r]{{2,}}){new_line_re}(?P<s_dash>-{{2,}}){new_line_re}')
    triggers = re.finditer(section_re, dcstring)
    matches = [(trigger.groupdict()['s_name'], trigger.span()) for trigger in triggers if
                len(trigger.groupdict()['s_name']) == len(trigger.groupdict()['s_dash'])] + \
              [(None, (len(dcstring), None))]
    sections = [m[0] for m in matches]
    starts = ends = 0
    if section in sections:
        i = sections.index(section)
        starts = matches[i][1][0]
        ends = matches[i + 1][1][0]

    if _return_section:
        return dcstring[starts:ends]
    else:
        return dcstring[:starts] + dcstring[ends:]


def rm_parameter(dcstring, parameter):
    """
    remove a parameters documentation from a function docstring
    """
    paramatches = _get_paramatches(dcstring)
    start = end = 0
    for p in paramatches:
        if parameter == p.groupdict()['paraname']:
            start = re.search(p[0], dcstring).span()[0]
            try:
                end = dcstring.find(next(paramatches)[0])
            except(StopIteration):
                end = len(re.sub(new_line_re + '$', '', dcstring))

    return dcstring[0:start] + dcstring[end:]


def get_parameter(dcstr):
    """
    returns the list of parameters and their defaults, documented in a docstrings Parameters section
    """
    paramatches = _get_paramatches(dcstr)
    return [(p.groupdict()['paraname'], p.groupdict()['paradefaults']) for p in paramatches]


def _get_paramatches(dcstr):
    parastr = rm_section(dcstr, 'Parameters', _return_section=True)
    match_re = f"{new_line_re}(?P<paraname>[\S]+) : [^\n\r]*(default (?P<paradefaults>[^\n\r]*))?"
    return re.finditer(match_re, parastr)


def parse_func_dcstrings(m_paths):
    func_dict = {}
    for m in m_paths:
        with open(m) as f:
            lines = f.readlines()
        module_ast = ast.parse(''.join(lines))
        funcs = [node for node in module_ast.body if isinstance(node, ast.FunctionDef)]
        for func in funcs:
            dcstr = ast.get_docstring(func)
            if func.name[0] == '_' or (dcstr is None):
                continue
            dcstr = rm_section(dcstr, 'Returns')
            dcstr = rm_parameter(dcstr, 'data')
            dcstr = rm_parameter(dcstr, 'flagger')
            parameters = get_parameter(dcstr)
            parameters = [f"{p[0]}={p[1]}" if p[1] else p[0] for p in parameters]
            signature = f"def {func.name}({', '.join(parameters)}):"
            # get @register module registration if present
            reg_module = None
            r = [d for d in func.decorator_list if d.func.id == 'register']
            if r:
                rm = [kw.value.s for kw in r[0].keywords if kw.arg == 'module']
                if rm:
                    reg_module = rm[0]

            func_dict[f"{os.path.splitext(os.path.basename(m))[0]}.{func.name}"] = (signature, dcstr, reg_module)

    return func_dict



def parse_module_dcstrings(m_paths):
    mod_dict = {}
    for m in m_paths:
        with open(m) as f:
            lines = f.readlines()

        mod_docstr = ast.get_docstring(ast.parse(''.join(lines)))
        mod_dict[f"{os.path.splitext(os.path.basename(m))[0]}"] = mod_docstr or ''
    return mod_dict


def make_doc_module(targetpath, func_dict, doc_mod_structure):
    for doc_mod in [d for d in doc_mod_structure.keys() if not re.search('_dcstring$', d)]:
        with open(os.path.join(targetpath, f"{doc_mod}.py"), 'w+') as f:
            mod_string = ['"""\n' + doc_mod_structure[doc_mod + '_dcstring'] + '\n"""']
            mod_funcs = doc_mod_structure[doc_mod]
            for func in mod_funcs:
                mod_string.append(func_dict[func][0])
                mod_string.append('    """')
                # indent the docstring:
                indented_doc_string = '\n'.join([f"    {l}" for l in func_dict[func][1].splitlines()])
                mod_string.append(indented_doc_string)
                mod_string.append('    """')
                mod_string.append('    pass')
                mod_string.append('')
                mod_string.append('')
            f.write('\n'.join(mod_string))

    with open(os.path.join(targetpath, 'module_dict.pkl'), 'wb+') as file:
        pickle.dump(doc_mod_structure, file)

    return 0


@click.command()
@click.option(
    "-p", "--pckpath", type=str,  required=True, default="saqc/funcs",
    help="Relative path to the package to be documented (relative to sphinx root)."
)
@click.option(
    "-t", "--targetpath", type=str,  required=True, default="docs/intro_modules",
    help="Output folder path (relative to sphinx root). Will be overridden if already existent."
)
@click.option(
    "-sr", "--sphinxroot", type=str,  required=True, default='..', help="Relative path to the sphinx root."
)
@click.option(
    "-mo", "--mode", type=str,  required=True, default='intro_doc', help="either 'intro_doc' or 'module_doc'."
)

def main(pckpath, targetpath, sphinxroot, mode):
    root_path = os.path.abspath(sphinxroot)
    pkg_path = os.path.join(root_path, pckpath)
    targetpath = os.path.join(root_path, targetpath)
    modules = []
    # collect modules
    for _, modname, _ in pkgutil.walk_packages(path=[pkg_path], onerror=lambda x: None):
        modules.append(modname)

    # clear target dir
    if os.path.isdir(targetpath):
        shutil.rmtree(targetpath)
    os.makedirs(targetpath, exist_ok=True)

    # parse all the functions
    module_paths = [os.path.join(pkg_path, f'{m}.py') for m in modules]
    mod_dict = parse_module_dcstrings(module_paths)
    func_dict = parse_func_dcstrings(module_paths)
    if mode == 'intro_doc':
        make_doc_module(targetpath, func_dict, doc_mod_structure)
    if mode == 'registered_doc':
        doc_struct = {}
        for dm in func_dict.keys():
            module = func_dict[dm][2]
            if module:
                if module in doc_struct.keys():
                    doc_struct[module].append(dm)
                else:
                    doc_struct[module] = [dm]
                    doc_struct[module + '_dcstring'] = mod_dict[module]
        make_doc_module(targetpath, func_dict, doc_struct)
    if mode == 'module_doc':
        doc_struct = {m:[] for m in modules}
        for dm in func_dict.keys():
            module = re.search('([^ .]*)\.[^ ]*$', dm).group(1)
            doc_struct[module].append(dm)
        make_doc_module(targetpath, func_dict, doc_struct)


if __name__ == "__main__":
    main()