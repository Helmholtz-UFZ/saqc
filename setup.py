# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

from distutils.util import convert_path

from setuptools import find_packages, setup

# read the version string from saqc without importing it. See the
# link for a more detailed description of the problem and the solution
# https://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
vdict = {}
version_fpath = convert_path("saqc/version.py")
with open(version_fpath) as f:
    exec(f.read(), vdict)
version = vdict["__version__"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="saqc",
    version=version,
    author="Bert Palm, David Schaefer, Florian Gransee, Peter Luenenschloss",
    author_email="david.schaefer@ufz.de",
    description="A timeseries data quality control and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.7",
    install_requires=[
        "Click",
        "dtw",
        "matplotlib>=3.4",
        "numba",
        "numpy",
        "outlier-utils",
        "pyarrow",
        "pandas>=1.2,<1.5",
        "scikit-learn",
        "scipy",
        "typing_extensions",
    ],
    license_files=("LICENSE.md", "LICENSES/GPL-3.0-or-later.txt"),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
