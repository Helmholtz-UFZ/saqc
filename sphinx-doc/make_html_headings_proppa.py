import os
import click
import re
import pickle


@click.command()
@click.option(
    "-b",
    "--buildpath",
    type=str,
    required=True,
    default="sphinx-doc/_build/html/_api",
    help="Relative path to the html api files to be manipulated (relative to sphinx root).",
)
@click.option(
    "-sr",
    "--sphinxroot",
    type=str,
    required=True,
    default="..",
    help="Relative path to the sphinx root.",
)
@click.option(
    "-p",
    "--pckpath",
    type=str,
    required=True,
    default="docs/doc_modules/func_modules",
    help="Relative path to the documented package (relative to sphinx root).",
)
def main(buildpath, sphinxroot, pckpath):
    root_path = os.path.abspath(sphinxroot)
    buildpath = os.path.join(root_path, buildpath)
    pckpath = os.path.join(root_path, pckpath)
    files = os.listdir(buildpath)
    # gather all files from the doc module
    files = [f for f in files if re.search("^docs\.", f)]
    with open(os.path.join(pckpath, "module_dict.pkl"), "rb") as file_:
        doc_mod_structure = pickle.load(file_)

    for key in doc_mod_structure.keys():
        # search for all function files assigned to the module
        mod_f = [f for f in files if re.search(f"(^|[.]){key}\.[^.]*\.html", f)]
        for file_ in mod_f:
            parts = file_.split(".")
            func = parts[-2]
            module_domain = ".".join(parts[:-2])

            with open(os.path.join(buildpath, file_), "r") as wf:
                code = wf.read()

            old_domain_str = f'<code class="sig-prename descclassname">{module_domain}'
            new_domain = [
                f.split(".")[0]
                for f in doc_mod_structure[key]
                if f.split(".")[1] == func
            ][0]
            new_domain_str = f'<code class="sig-prename descclassname">{new_domain}'
            code = code.replace(old_domain_str, new_domain_str)
            with open(os.path.join(buildpath, file_), "w+") as wf:
                wf.write(code)


if __name__ == "__main__":
    main()
