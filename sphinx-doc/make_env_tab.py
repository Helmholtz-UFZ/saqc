import ast
import os
import click
import pkgutil
import shutil
import re
from collections import OrderedDict
import pickle


@click.command()
@click.option(
    "-p",
    "--envpath",
    type=str,
    required=True,
    default="saqc/constants.py",
    help="Relative path to the module containing the env dict (relative to sphinx root).",
)
@click.option(
    "-t",
    "--targetpath",
    type=str,
    required=True,
    default="sphinx-doc/environment",
    help="Output path to contain configEnv.rst (relative to sphinx root).",
)
@click.option(
    "-sr",
    "--sphinxroot",
    type=str,
    required=True,
    default="..",
    help="Relative path to the sphinx root.",
)
def main(envpath, targetpath, sphinxroot):
    root_path = os.path.abspath(sphinxroot)
    source_path = os.path.join(root_path, envpath)
    target_path = os.path.join(root_path, targetpath)

    with open(source_path) as f:
        lines = f.readlines()

    # get ENV definition linenumber
    lino_st = None
    lino_end = None
    nodes = ast.parse("".join(lines))
    for node in nodes.body:
        if lino_st:
            lino_end = node.lineno
            break
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "ENVIRONMENT":
                    lino_st = node.lineno

    env_dict = lines[slice(lino_st, lino_end)]
    entry = 0
    val_dict = {}
    doc_dict = {}
    for line in env_dict:
        val = re.match(' *(?P<Key>".*"):', line)
        doc = re.match(" *#(?P<Comment>.*)", line)
        if val:
            if len(val.groups()) > 0:
                entry += 1
                val_dict.update({entry: val["Key"]})
        if doc:
            if len(doc.groups()) > 0:
                doc_dict.update({entry: doc_dict.get(entry, "") + doc["Comment"]})

    # make heading
    heading = "Config file Environment"
    lines = []
    lines += [heading, len(heading) * "=", ""]
    # make table directive
    tab = " " * 3
    new_row = tab + "* - "
    new_col = tab + " " + " - "
    # directive:
    lines += [".. list-table::", tab + ":header-rows: 1", ""]
    # table header:
    lines += [new_row + "Env-Value"]
    lines += [new_col + "Env-Doc"]
    # table body:
    for k in val_dict.keys():
        lines += [new_row + val_dict[k]]
        lines += [new_col + doc_dict.get(k, "")]

    with open(os.path.join(target_path, "configEnv.rst"), "w") as f:
        for s in lines:
            f.write(s + "\n")


if __name__ == "__main__":
    main()
