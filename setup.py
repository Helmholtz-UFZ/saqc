# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later

import versioneer
from setuptools import find_packages, setup

# read the version string from saqc without importing it. See the
# link for a more detailed description of the problem and the solution
# https://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
with open("README.md", "r") as fh:
    long_description = fh.read()

v = versioneer.get_versions()

if v["error"]:
    raise RuntimeError(v["error"])

if v["dirty"]:
    raise ValueError(
        f"The repository you build is dirty. Please commit changes first {v}."
    )

setup(
    name="saqc",
    version=versioneer.get_version(),  # keep this line as it is
    cmdclass=versioneer.get_cmdclass(),  # keep this line as it is
    author="David Schaefer, Bert Palm, Peter Luenenschloss",
    author_email="david.schaefer@ufz.de",
    description="A timeseries data quality control and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.11",
    install_requires=[
        "click",
        "docstring_parser",
        "fancy-collections==0.3.0",
        "fastdtw==0.3.4",
        "matplotlib",
        "numpy",
        "pyarrow",
        "pymoo",
        "pandas",
        "polars",
        "pydantic",
        "scikit-learn",
        "scipy",
        "pydantic",
        "pywavelets",
    ],
    license_files=("LICENSE.md", "LICENSES/GPL-3.0-or-later.txt"),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
