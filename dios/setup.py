# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import subprocess

from setuptools import find_packages, setup

with open("Readme.md", "r") as fh:
    long_description = fh.read()

cmd = "git describe --tags --always --dirty"
version = (
    subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE)
    .stdout.decode()
    .strip()
)
print(f"git version: {version}")
# if '-dirty' in version:
#     print("Do not make a version from a dirty repro. Exiting now")
#     exit(1)
txt = "enter version\n>"
version = input(txt)

setup(
    name="dios",
    version=version,
    author="Bert Palm",
    author_email="bert.palm@ufz.de",
    description="Dictionary of Series - a kind of pandas extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm/dios",
    packages=["dios"],
    install_requires=[
        "pandas",
    ],
    license="GPLv3",
)
