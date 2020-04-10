from setuptools import setup, find_packages
import saqc

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="saqc",
    version=saqc.__version__,
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="david.schaefer@ufz.de",
    description="Data quality checking and processing tool/framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "numba",
        "matplotlib",
        "click",
        "pyarrow",
        "python-intervals",
        "astor",
    ],
    license="GPLv3",
    entry_points={"console_scripts": ["saqc=saqc.__main__:main"],},
)
