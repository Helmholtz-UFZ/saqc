import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="saqc",
    version="1.0.0",
    author="Bert Palm, David Schaefer, Peter Luenenschloss, Lennard Schmidt",
    author_email="bert.palm@ufz.de, david.schaefer@ufz.de, peter.luenenschloss@ufz.de, lennart.schmidt@ufz.de",
    description="Data quality checking and processing framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ufz.de/rdm-software/saqc",
    packages=["saqc"],
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
    entry_points = {
        'console_scripts': ['saqc=saqc.__main__:main'],
    }
)
