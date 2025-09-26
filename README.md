<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

<br>
<div align="center">
  <img src="https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/images/representative/SaQCLogo.png" width="300">
</div>

-----------------
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

#  SaQC: System for automated Quality Control

SaQC is a flexible framework for quality control of time series data.
It offers a growing collection of algorithms and methods to analyze, annotate, and process time series, while supporting end-to-end metadata enrichment.

SaQC provides multiple user interfaces to fit different workflows:

- A Python API for programmatic integration
- A command-line interface with a text-based configuration system
- A [Galaxy Tool](https://usegalaxy.eu/root?tool_id=saqc) for integration into data analysis pipelines
- A web-based [user interface](https://web.app.ufz.de/saqc-config-app/) for interactive exploration

Designed with the needs of data professionals in mind — ranging from sensor engineers to domain experts and data scientists — SaQC
helps improve and standardize the quality of data products across disciplines.

For an up-to-date overview of features, usage patterns, system components, and customization options, please refer to the
[online documentation](https://rdm-software.pages.ufz.de/saqc/index.html)

## Installation

`SaQC` is available on the Python Package Index ([PyPI](https://pypi.org/)) and
can be installed using [pip](https://pip.pypa.io/en/stable/):
```sh
python -m pip install saqc
```
Additionally `SaQC` is available via conda and can be installed with:

```sh
conda create -c conda-forge -n saqc saqc
```

For more details, see the [installation guide](https://rdm-software.pages.ufz.de/saqc/gettingstarted/InstallationGuide.html).


## Usage

SaQC can be used as a command-line application driven by a text-based configuration, or as a Python module with a consistent and simple API.

### SaQC as a command-line application

The command-line application is controlled through a semicolon-separated configuration file. This file lists the dataset variables
along with the routines to inspect, quality control, and process them.

An example configuration file can be found [here](https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/data/config.csv)

```
varname    ; test
#----------; ---------------------------------------------------------------------
SM2        ; align(freq="15Min")
'SM(1|2)+' ; flagMissing()
SM1        ; flagRange(min=10, max=60)
SM2        ; flagRange(min=10, max=40)
SM2        ; flagZScore(window="30d", thresh=3.5, method='modified', center=False)
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
    --config https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/data/config.csv \
    --data https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/data/data.csv \
    --outfile saqc_test.csv
```

### SaQC as a python module

When used as a Python module, SaQC provides a consistent and simple API for programmatic access. This makes it straightforward
to define, execute, and customize quality control routines directly within Python, and to integrate SaQC into data analysis workflows and scripts.
The following snippet implements the same configuration given above through the Python-API:

```python
import pandas as pd
from saqc import SaQC

data = pd.read_csv(
    "https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/data/data.csv",
    index_col=0, parse_dates=True,
)

qc = SaQC(data=data)
qc = (qc
      .align("SM2", freq="15Min")
      .flagMissing("SM(1|2)+", regex=True)
      .flagRange("SM1", min=10, max=60)
      .flagRange("SM2", min=10, max=40)
      .flagZScore("SM2", window="30d", thresh=3.5, method='modified', center=False)
      .flagGeneric(field=["SM1", "SM2"], target="Dummy", func=lambda x, y: (isflagged(x) | isflagged(y))))
```

A more detailed description of the Python API is available in the
[respective section](https://rdm-software.pages.ufz.de/saqc/gettingstarted/TutorialAPI.html)
of the documentation.

## Get involved

### Contributing
You found a bug or you want to suggest new features? Please refer to our [contributing guidelines](CONTRIBUTING.md) to see how you can contribute to SaQC.

### User support
If you need help or have questions, send us an email to [saqc-support@ufz.de](mailto:saqc-support@ufz.de)

## Copyright and License
Copyright(c) 2021, [Helmholtz-Zentrum für Umweltforschung GmbH -- UFZ](https://www.ufz.de). All rights reserved.

- Documentation: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>
- Source code: [GNU General Public License 3](https://www.gnu.org/licenses/gpl-3.0.html)

For full details, see [LICENSE](LICENSE.md).

## Publications
> Lennart Schmidt, David Schäfer, Juliane Geller, Peter Lünenschloss, Bert Palm, Karsten Rinke, Corinna Rebmann, Michael Rode, Jan Bumberger, System for automated Quality Control (SaQC) to enable traceable and reproducible data streams in environmental science, Environmental Modelling & Software, 2023, 105809, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2023.105809. (https://www.sciencedirect.com/science/article/pii/S1364815223001950)

## How to cite SaQC
If SaQC is advancing your research, please cite as:

> Schäfer, David, Palm, Bert, Lünenschloß, Peter, Schmidt, Lennart, & Bumberger, Jan. (2023). System for automated Quality Control - SaQC (2.3.0). Zenodo. https://doi.org/10.5281/zenodo.5888547

or

> Lennart Schmidt, David Schäfer, Juliane Geller, Peter Lünenschloss, Bert Palm, Karsten Rinke, Corinna Rebmann, Michael Rode, Jan Bumberger, System for automated Quality Control (SaQC) to enable traceable and reproducible data streams in environmental science, Environmental Modelling & Software, 2023, 105809, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2023.105809. (https://www.sciencedirect.com/science/article/pii/S1364815223001950)

-----------------

<a href="https://www.ufz.de/index.php?en=33573">
    <img src="https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/images/representative/UFZLogo.png" width="400"/>
</a>

<a href="https://www.ufz.de/index.php?en=45348">
    <img src="https://git.ufz.de/rdm-software/saqc/raw/main/docs/resources/images/representative/RDMLogo.png" align="right" width="220"/>
</a>
