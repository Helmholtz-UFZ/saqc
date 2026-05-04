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
        "click>=8.3.3",
        "docstring_parser>=0.18",
        "fancy-collections==0.3.0",
        "fastdtw==0.3.4",
        "matplotlib>=3.10.9",
        "numpy<=2.2.6",
        "outlier-utils==0.0.5",
        "pyarrow>=24.0.0",
        "pymoo>=0.6.1.6",
        "pandas>=3.0.2",
        "pydantic>=2.13.3",
        "scikit-learn>=1.8.0",
        "scipy<=1.14.1",
        "typing_extensions>=4.15.0",
        "eval-type-backport>=0.3.1",
    ],
    license_files=("LICENSE.md", "LICENSES/GPL-3.0-or-later.txt"),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
