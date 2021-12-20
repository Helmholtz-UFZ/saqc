from setuptools import setup, find_packages
from distutils.util import convert_path

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
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="david.schaefer@ufz.de",
    description="Data quality checking and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7, <3.10",
    install_requires=[
        "Click==8.0.*",
        "dtw==1.4.*",
        "matplotlib>=3.4,<3.6",
        "numba==0.54.*",
        "numpy==1.20.*",
        "outlier-utils==0.0.3",
        "pyarrow==6.0.*",
        "pandas==1.3.*",
        "scikit-learn==1.0.*",
        "scipy==1.7.*",
        "typing_extensions==4.*",
        "seaborn==0.11.*",
    ],
    license_files=("LICENSE.md",),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
