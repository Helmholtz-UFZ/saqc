#! /usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import difflib
from pathlib import Path
from typing import List, Sequence

import click
import black


FUNCS_PATH = Path("saqc/funcs")
CORE_PATH = Path("saqc/core/core.py")
PYI_PATH = Path("saqc/core/core.pyi")


def isMultivariate(func: ast.FunctionDef):
    """
    check if a function is multivariate (i.e. sets the decorator paramater
    ``multivariate = True``)
    """
    for dec in func.decorator_list:
        if dec.func.id == "register":
            # NOTE: `multivariate` should be a keyword only attribute
            kwargs = {k.arg: k.value for k in dec.keywords}
            if "multivariate" in kwargs:
                return kwargs["multivariate"].value
    return False


def isTestFunction(func: ast.FunctionDef):
    """
    check whether a function is test function (i.e. decorated with ``@register``)
    """

    for dec in func.decorator_list:
        if isinstance(dec.func, ast.Name) and dec.func.id == "register":
            return True
    return False


def prepTestFunction(func: ast.FunctionDef) -> ast.FunctionDef:
    """
    - remove the function body by ``Ellipsis``
    - ensure the ``field`` type ``Union[str | Sequence[str]]``
    - change the return type to ``SaQC``
    - add the self parameter
    - add the additional parameters `target`, `flag`, `to_mask`
    """
    selfarg = ast.arg(arg="self", annotation=None, type_comment=None)

    add_args = {
        "target": ast.arg(
            arg="target", annotation=ast.Name("str | Sequence[str]"), type_comment=None
        ),
        "flag": ast.arg(
            arg="flag", annotation=ast.Name("ExternalFlag"), type_comment=None
        ),
        "to_mask": ast.arg(
            arg="to_mask", annotation=ast.Name("ExternalFlag"), type_comment=None
        ),
    }

    args = [a for a in func.args.args if a.arg not in ("data", "flags")]
    for a in args:
        if a.arg == "field":
            a.annotation = ast.Name("str | Sequence[str]")
        if a.arg in add_args:
            del add_args[a.arg]
    # add the additional keyword arguments
    func.args.args = [selfarg] + args + list(add_args.values())
    # add the default values for the additional keyword arguments
    func.args.defaults += [ast.Constant(value=None, kind=None)] * len(add_args)

    body = ast.Expr(value=ast.Constant(value=Ellipsis, kind=None))
    func.body = [body]

    func.decorator_list = []
    func.returns = ast.Name(id="SaQC")
    return func


def extractTestFunctions(module: Path) -> Sequence[ast.FunctionDef]:
    """
    get and prepare all test functions in ``module``
    """
    with open(module, "r") as f:
        content = f.read()

    mod = ast.parse(content)
    functions = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    sigs = []
    for function in functions:
        if isTestFunction(function):
            sigs.append(prepTestFunction(function))

    return sigs


def prepClass(cls: ast.ClassDef) -> ast.ClassDef:
    """
    - remove all magic method with exception of ``__init__`` and ``__getitem__``
    - remove all private methods (i.e. starting with at least on underscore)
    - replace method bodies with ``Ellipsis``
    """
    body = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [ast.Expr(value=ast.Constant(value=Ellipsis, kind=None))]
            body.append(node)
    cls.body = body
    return cls


def extractClasses(module: Path) -> List[ast.ClassDef]:
    """
    get and prepare all classes in ``module``
    """
    with open(module, "r") as f:
        content = f.read()
    mod = ast.parse(content)
    out = []
    for node in mod.body:
        if isinstance(node, ast.ClassDef):
            out.append(prepClass(node))
    return out


def toModule(nodes: List[ast.ClassDef]) -> ast.Module:
    imports = [
        ast.Import(names=[ast.alias(name="numpy", asname="np")]),
        ast.Import(names=[ast.alias(name="pandas", asname="pd")]),
        ast.ImportFrom(
            module="typing",
            names=[
                ast.alias(a)
                for a in (
                    "Sequence",
                    "Union",
                    "Tuple",
                    "Hashable",
                    "Any",
                    "Optional",
                    "Callable",
                )
            ],
            level=0,
        ),
        ast.ImportFrom(
            module="typing_extensions",
            names=[ast.alias(a) for a in ("Literal",)],
            level=0,
        ),
        ast.ImportFrom(
            module="dios",
            names=[ast.alias("DictOfSeries")],
            level=0,
        ),
        ast.ImportFrom(
            module="saqc.constants",
            names=[ast.alias(a) for a in ("UNFLAGGED", "BAD")],
            level=0,
        ),
        ast.ImportFrom(
            module="saqc.core.flags",
            names=[ast.alias("Flags")],
            level=0,
        ),
        ast.ImportFrom(
            module="saqc.core.translator",
            names=[ast.alias("Translator")],
            level=0,
        ),
        ast.ImportFrom(
            module="saqc.core.register",
            names=[ast.alias("FunctionWrapper")],
            level=0,
        ),
        ast.ImportFrom(
            module="scipy.spatial.distance",
            names=[ast.alias("pdist")],
            level=0,
        ),
        ast.ImportFrom(
            module="saqc.lib.types",
            names=[
                ast.alias(a)
                for a in (
                    "FreqString",
                    "CurveFitter",
                    "LinkageString",
                    "InterpolationString",
                    "ExternalFlag",
                    "GenericFunction",
                )
            ],
            level=0,
        ),
    ]

    return ast.Module(body=imports + nodes, type_ignores=[])


def printDiff(left: str, right: str):
    for line in difflib.ndiff(left.splitlines(), right.splitlines()):
        print(line)


@click.command()
@click.option(
    "-c", "--check", is_flag=True, default=False, help="check for changes only"
)
def main(check):
    classes = extractClasses(CORE_PATH)

    functions = []
    for module in sorted(FUNCS_PATH.glob("*.py")):
        functions.extend(extractTestFunctions(module))

    saqc = [cls for cls in classes if cls.name == "SaQC"][0]
    for func in functions:
        saqc.body.append(func)

    mod = toModule(classes)
    mod_string = ast.unparse(mod)
    out = black.format_str(mod_string, mode=black.FileMode(is_pyi=True))
    if check:
        if not PYI_PATH.exists():
            raise RuntimeError(f"the signature file '{PYI_PATH}' does not exist")
        with open(PYI_PATH, "r") as f:
            existing = f.read()
        if mod_string != ast.unparse(
            ast.parse(existing)
        ):  # to get over formatting differences
            printDiff(out, existing)
            raise RuntimeError(
                "the generated signatures have changed, please run this script again"
            )
    else:
        with open(PYI_PATH, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
