
<img src="sphinx-doc/ressources/images/Representative/UFZLogo.jpg" width="400"/>

<img src="sphinx-doc/ressources/images/Representative/RDMlogo.jpg" align="right" width="180"/>


# System for automated Quality Control (SaQC)

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
For a more detailed installion guide, see the [installation guide](https://rdm-software.pages.ufz.de/saqc/getting_started/InstallationGuide.html).

## Usage

`SaQC` is both, a command line application controlled by a text based configuration
and a python module with a simple API.

### SaQC as a command line application
The command line application is controlled by a semicolon-separated text
file listing the variables in the dataset and the routines to inspect,
quality control and/or process them. The content of such a configuration
could look like this:

```
varname    ; test
#----------;------------------------------------
SM2        ; shiftToFreq(freq="15Min")
SM2        ; flagMissing()
'SM(1|2)+' ; flagRange(min=10, max=60)
SM2        ; flagMad(window="30d", z=3.5)
```

As soon as the basic inputs, dataset and configuration file, are
prepared, `SaQC` is run with:
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

A more detailed description of the Python API is available in the 
[respective section](https://rdm-software.pages.ufz.de/saqc/getting_started/TutorialAPI.html)
of the documentation.
