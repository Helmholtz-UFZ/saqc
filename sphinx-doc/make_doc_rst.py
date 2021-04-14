import os
import click
import pkgutil
import ast
import shutil


def parse_imports(path):
    modules = []
    file = open(path)
    lines = file.readlines()
    for node in ast.iter_child_nodes(ast.parse("".join(lines))):
        if isinstance(node, ast.ImportFrom) | isinstance(node, ast.Import):
            modules += [x.name for x in node.names] + [
                x.asname for x in node.names if x.asname is not None
            ]
    file.close()
    return modules


@click.command()
@click.option(
    "-p",
    "--pckpath",
    type=str,
    required=True,
    default="saqc/funcs",
    help="Relative path to the package to be documented (relative to sphinx root).",
)
@click.option(
    "-t",
    "--targetpath",
    type=str,
    required=True,
    default="sphinx-doc/internal_doc_rst",
    help="Output folder path (relative to sphinx root). Will be overridden if already existent.",
)
@click.option(
    "-sr",
    "--sphinxroot",
    type=str,
    required=True,
    default="..",
    help="Relative path to the sphinx root.",
)
def main(pckpath, targetpath, sphinxroot):
    root_path = os.path.abspath(sphinxroot)
    targetpath = os.path.join(root_path, targetpath)
    pkg_path = os.path.join(root_path, pckpath)
    modules = []
    for _, modname, _ in pkgutil.walk_packages(path=[pkg_path], onerror=lambda x: None):
        modules.append(modname)

    emptyline = [""]

    # clear target directory:
    if os.path.isdir(targetpath):
        shutil.rmtree(targetpath)
    os.mkdir(targetpath)

    for module in modules:
        imports = parse_imports(os.path.join(pkg_path, f"{module}.py"))
        skiplist = [f"\t:skip: {k}" for k in imports]
        section = [module] + ["=" * len(module)]
        automodapi_directive = [
            ".. automodapi:: " + pckpath.replace("/", ".") + "." + module
        ]
        no_heading = [f"\t:no-heading:"]
        to_write = (
            emptyline
            + section
            + emptyline
            + automodapi_directive
            + skiplist
            + no_heading
        )
        to_write = "".join([f"{k}\r\n" for k in to_write])
        with open(os.path.join(targetpath, f"{module}.rst"), "w+") as f:
            f.write(to_write)


if __name__ == "__main__":
    main()
