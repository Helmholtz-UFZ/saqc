<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

<br>
<div align="center">
  <img src="https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/images/representative/SaQCLogo.png" width="300">
</div>

-----------------
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

#  SaQC: System for automated Quality Control

Anomalies and errors are the rule not the exception when working with 
time series data. This is especially true, if such data originates
from in-situ measurements of environmental properties. 
Almost all applications, however, implicily rely on data, that complies
with some definition of 'correct'. 
In order to infer reliable data products and tools, there is no alternative
to quality control. SaQC provides all the building blocks to comfortably
bridge the gap between 'usually faulty' and 'expected to be corrected' in 
a accessible, consistent, objective and reproducible way.

For a (continously improving) overview of features, typical usage patterns,
the specific system components and how to customize `SaQC` to your specific
needs, please refer to our
[online documentation](https://rdm-software.pages.ufz.de/saqc/index.html).


## Installation

SaQC is available on the Python Package Index ([PyPI](https://pypi.org/)) and
can be installed using [pip](https://pip.pypa.io/en/stable/):
```sh
python -m pip install saqc
```
For a more detailed installion guide, see the [installation guide](https://rdm-software.pages.ufz.de/saqc/gettingstarted/InstallationGuide.html).

## Usage

`SaQC` is both, a command line application controlled by a text based configuration
and a python module with a simple API.

### SaQC as a command line application
The command line application is controlled by a semicolon-separated text
file listing the variables in the dataset and the routines to inspect,
quality control and/or process them. The content of such a configuration
could look like [this](https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/data/config.csv):

```
varname    ; test
#----------; ---------------------------------------------------------------------
SM2        ; shift(freq="15Min")
'SM(1|2)+' ; flagMissing()
SM1        ; flagRange(min=10, max=60)
SM2        ; flagRange(min=10, max=40)
SM2        ; flagMAD(window="30d", z=3.5)
Dummy      ; flagGeneric(field=["SM1", "SM2"], func=(isflagged(x) | isflagged(y)))
```

As soon as the basic inputs, dataset and configuration file, are
prepared, run `SaQC`:
```sh
saqc \
    --config PATH_TO_CONFIGURATION \
    --data PATH_TO_DATA \
    --outfile PATH_TO_OUTPUT
```

A full `SaQC` run against provided example data can be invoked with:
```sh
saqc \
    --config https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/data/config.csv \
    --data https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/data/data.csv \
    --outfile saqc_test.csv
```

### SaQC as a python module

The following snippet implements the same configuration given above through
the Python-API:

```python
import pandas as pd
from saqc import SaQC

data = pd.read_csv(
    "https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/data/data.csv",
    index_col=0, parse_dates=True,
)

saqc = SaQC(data=data)
saqc = (saqc
        .shift("SM2", freq="15Min")
        .flagMissing("SM(1|2)+", regex=True)
        .flagRange("SM1", min=10, max=60)
        .flagRange("SM2", min=10, max=40)
        .flagMAD("SM2", window="30d", z=3.5)
        .flagGeneric(field=["SM1", "SM2"], target="Dummy", func=lambda x, y: (isflagged(x) | isflagged(y))))
```

A more detailed description of the Python API is available in the 
[respective section](https://rdm-software.pages.ufz.de/saqc/gettingstarted/TutorialAPI.html)
of the documentation.

## Changelog
All notable changes to this project will be documented in [CHANGELOG.md](CHANGELOG.md).

## Get involved

### Contributing
You found a bug or you want to suggest some cool features? Please refer to our [contributing guidelines](CONTRIBUTING.md) to see how you can contribute to SaQC.

### User support
If you need help or have a question, you can use the SaQC user support mailing list: [saqc-support@ufz.de](mailto:saqc-support@ufz.de)

## Copyright and License
Copyright(c) 2021, [Helmholtz-Zentrum für Umweltforschung GmbH -- UFZ](https://www.ufz.de). All rights reserved.

- Documentation: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>
- Source code: [GNU General Public License 3](https://www.gnu.org/licenses/gpl-3.0.html)

For full details, see [LICENSE](LICENSE.md).

## Acknowledgements
...

## Publications
coming soon...

## How to cite SaQC
If SaQC is advancing your research, please cite as:

> Schäfer, David, Palm, Bert, Lünenschloß, Peter, Schmidt, Lennart, & Bumberger, Jan. (2023). System for automated Quality Control - SaQC (2.3.0). Zenodo. https://doi.org/10.5281/zenodo.5888547

-----------------

<a href="https://www.ufz.de/index.php?en=33573">
    <img src="https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/images/representative/UFZLogo.png" width="400"/>
</a>

<a href="https://www.ufz.de/index.php?en=45348">
    <img src="https://git.ufz.de/rdm-software/saqc/raw/develop/docs/resources/images/representative/RDMLogo.png" align="right" width="220"/>
</a>
