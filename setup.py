from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="saqc",
    version="1.4",
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="david.schaefer@ufz.de",
    description="Data quality checking and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7, <3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "numba",
        "matplotlib",
        "click",
        "pyarrow",
        "typing_extensions",
        "outlier-utils",
        "dtw",
        "PyWavelets",
        "mlxtend",
    ],
    license_files=("LICENSE.md",),
    entry_points={
        "console_scripts": ["saqc=saqc.__main__:main"],
    },
)
