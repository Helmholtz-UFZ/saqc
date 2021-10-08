[![pipeline status](https://git.ufz.de/rdm-software/saqc/badges/develop/pipeline.svg)](https://git.ufz.de/rdm-software/saqc/-/commits/develop)

# System for automated Quality Control (SaQC)

Quality Control of numerical data requires a significant amount of
domain knowledge and practical experience. Finding a robust setup of
quality tests that identifies as many suspicious values as possible, without
removing valid data, is usually a time-consuming endeavor,
even for experts.

SaQC is both, a Python framework and a command line application, that
addresses the exploratory nature of quality control by offering a
continuously growing number of quality check routines through a flexible
and simple configuration system.

Below its user interface, SaQC is highly customizable and extensible.
A modular structure and well-defined interfaces make it easy to extend
the system with custom quality checks. Furthermore, even core components like
the flagging scheme are exchangeable.

![SaQC Workflow](ressources/images/readme_image.png "SaQC Workflow")

## Why?
During the implementation of data workflows in environmental sciences,
our experience shows a significant knowledge gap between the people
collecting data and those responsible for the processing and the
quality-control of these datasets.
While the former usually have a solid understanding of the underlying
physical properties, measurement principles and the resulting errors,
the latter are mostly software developers with expertise in
data processing.

The main objective of SaQC is to bridge this gap by allowing both
parties to focus on their strengths: The data collector/owner should be
able to express his/her ideas in an easy way, while the actual
implementation of the algorithms is left to the respective developers.


## How?

`SaQC` is both a command line application controlled by a text based configuration file and a python
module with a simple API.

While a good (but still growing) number of predefined and highly configurable
[functions](docs/FunctionIndex.md) are included and ready to use, SaQC
additionally ships with a python based
[extension language](docs/GenericFunctions.md) for quality and general
purpose data processing.

For a more specific round trip to some of SaQC's possibilities, we refer to
our [GettingStarted](docs/GettingStarted.md).


### SaQC as a command line application
Most of the magic is controlled by a
[semicolon-separated text file](saqc/docs/ConfigurationFiles.md) listing the variables of the
dataset and the routines to inspect, quality control and/or process them.
The content of such a configuration could look like this:

```
varname    ; test
#----------;------------------------------------
SM2        ; shiftToFreq(freq="15Min")
SM2        ; flagMissing()
'SM(1|2)+' ; flagRange(min=10, max=60)
SM2        ; flagMad(window="30d", z=3.5)
```

As soon as the basic inputs, a dataset and the configuration file are
prepared, running SaQC is as simple as:
```sh
saqc \
    --config path_to_configuration.txt \
    --data path_to_data.csv \
    --outfile path_to_output.csv
```

### SaQC as a python module

The following snippet implements the same configuration given above through
the Python-API:

```python
import numpy as np
from saqc import SaQC

saqc = (SaQC(data)
        .shiftToFreq("SM2", freq="15Min")
        .flagMissing("SM2")
        .flagRange("SM(1|2)+", regex=True, min=10, max=60)
        .flagMad("SM2", window="30d", z=3.5))

data, flags = saqc.getResult()
```

## Installation

### Python Package Index
SaQC is available on the Python Package Index ([PyPI](https://pypi.org/)) and
can be installed using [pip](https://pip.pypa.io/en/stable/):
```sh
python -m pip install saqc
```
For a more detailed installion guide, see [GettingStarted](docs/GettingStarted.md).

### Anaconda
Currently we don't provide pre-build conda packages but the installing of `SaQC`
using the [conda package manager](https://docs.conda.io/en/latest/) is
straightforward:
1. Create an anaconda environment including all the necessary dependencies with:
   ```sh
   conda env create -f environment.yml
   ```
2. Load the freshly created environment with:
   ```sh
   conda activate saqc
   ```

### Manual installation

The latest development version is directly available from the
[gitlab](https://git.ufz.de/rdm-software/saqc) server of the
[Helmholtz Center for Environmental Research](https://www.ufz.de/index.php?en=33573).
More details on how to install using the gitlab server are available
[here](docs/GettingStarted.md).

### Python version
The minimum Python version required is 3.7.

## License
Copyright(c) 2019,
Helmholtz Centre for Environmental Research - UFZ.
All rights reserved.

The "System for Automated Quality Control" is free software. You can
redistribute it and/or modify it under the terms of the GNU General
Public License as published by the free Software Foundation either
version 3 of the License, or (at your option) any later version. See the
[license](LICENSE.txt) for details.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
