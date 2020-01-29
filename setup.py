from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="saqc",
    version="1.1.0",
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
        "scikit-learn==0.21.2",
        "numba",
        "matplotlib",
        "click",
        "pyarrow",
        "python-intervals",
    ],
    license="GPLv3",
    entry_points={"console_scripts": ["saqc=saqc.__main__:main"],},
)
