import os
import click
import time


@click.command()
@click.option(
    "-src",
    "--source",
    type=str,
    required=True,
    default="sphinxdoc.coredoc.SaQC",
)
@click.option(
    "-trg",
    "--target",
    type=str,
    required=True,
    default="saqc.SaQC",
)
@click.option(
    "-br",
    "--builddir",
    type=str,
    required=True,
    default="_build",
    help="Relative path to the build dir.",
)
def main(source, target, builddir):
    builddir = os.path.abspath(builddir)
    apidir = os.path.join(builddir, os.path.normpath("html/_api"))
    os.remove(os.path.join(apidir, target + ".html"))
    with open(os.path.join(apidir, source + ".html"), "r") as f:
        APIstring = f.read()
    # APIstring = APIstring.replace('sphinxdoc.coredoc.core', 'saqc')

    APIstring = APIstring.replace(source, target)
    with open(os.path.join(apidir, target + ".html"), "w+") as f:
        f.write(APIstring)


if __name__ == "__main__":
    main()
