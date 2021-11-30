from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="saqc",
    version="2.0.0",
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="david.schaefer@ufz.de",
    description="Data quality checking and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7, <3.10",
    install_requires=[
        "numpy==1.20.*",
        "pandas==1.3.*",
        "scipy==1.7.*",
        "scikit-learn==1.0.*",
        "numba==0.54.*",
        "matplotlib==3.4.*",
        "Click==8.0.*",
        "pyarrow==4.0.*",
        "typing_extensions==3.10.*",
        "outlier-utils==0.0.3",
        "dtw==1.4.*",
        "seaborn==0.11.*",
    ],
    license_files=("LICENSE.md",),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
