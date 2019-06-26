import subprocess

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="saqc",
    version=subprocess.check_output('git describe --tags --always'.split()).decode().strip(),
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="bert.palm@ufz.de, david.schaefer@ufz.de, peter.luenenschloss@ufz.de, lennart.schmidt@ufz.de",
    description="automated quality assurance and control tool",
    long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm/saqc",
    packages=['saqc'],
    license='RDM Team - UFZ',
)
