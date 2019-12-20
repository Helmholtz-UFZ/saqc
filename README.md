# System for automated Quality Control (SaQC)

Quality Control of numerical data is an profoundly knowledge and experience
based activity. Finding a robust setup is usually a time consuming and dynamic
endeavor, even for an experienced data expert.

SaQC addresses the iterative and explorative characteristics of quality control
with its extensive setup and configuration possibilities and a python based
extension language. To make the system flexible, many aspects of the quality
checking process, like

+ test parametrization
+ test evaluation and 
+ test exploration 

are easily configurable with plain text files.

Below its userinterface, SaQC is, thus, highly customizable and extensible.
Well defined interfaces allow the extension with new quality check routines.
Additionally, many core components, like the flagging scheme, are replaceable.

---

## Why?
When it comes to the implementation of data workflows in the environmental
sciences, our experience in (research) data management revealed a significant
knowledege gap between the people collecting often large amounts of
(environmental) data, and the persons responsible for the processing and the
quality asssurence of these datasets.
While the former usually have a good understanding of the underlying measurement
principles, potential noise sources overlaying the actual signal and the
expected characteristics of the dataset, the latter are mostly software
developers with a good knowledge on how to implement data flows.

The main objective of SaQC is therefore to bridge this gap by allowing both
parties to concentrate on their strengths: the data collector/owner should be
able to express her ideas in an easy and succint way while the actual 
implementation of the data processing and quality checking is left to the 
respective experts.


## How?
The most import aspect of SaQC, the general configuration of the system,
is text-based. All the magic takes place in a semicolon-separated table file
listing the variables within the dataset to inspect, quality control and/or
modify. 

While a good (but still growing) number of predifined and heighly configurable
[functions](docs/FunctionDescriptions.md) are included and ready to use, SaQC
additionally ships with a python based extension language. The, let's call it
slightly exxagerated, domain specific language (DSL), allows to define (more
or less simple) tests to be written directly within in the configuration. The 
idea is, that many more complex datasets carry inherent physical and 
technical relationsships (like "if the variables indicating the health of 
an active cooling solution drops, the values of variable 'y' are useless"), that
are way easier to express in text than in code. 

For a more specific round trip to some of SaQC's possibilities, please refer to
our [HowTo](docs/GettingStarted.md).

## Installation

### pip
SaQC is available on the Python Package Index ([PyPI](https://pypi.org/)) and
can be installed using [pip](https://pip.pypa.io/en/stable/):
```sh
python -m pip install saqc
```

### Manual installation
The latest development version is directly available from the
[gitlab](https://git.ufz.de/rdm-software/saqc) server of the
[Helmholtz Center for Environmental Research](https://www.ufz.de/index.php?en=33573). 
   
   
## Usage
### Command line interface (CLI)
SaQC provides a basic CLI 

### Integration into the 

### User Defined Test
User defined tests allow to specify simple quality checks directly within the
configuration.
#### Specification
- Test name: `generic`
- The parameter 'func' followed by an expression needs to be given
- Example: generic, `{func: (thisvar > 0) & ismissing(othervar)}`
#### Restrictions
- only the operators and functions listed below are available
- all checks need to be conditional expression and have to return an array of boolean values, 
  all other expressions are rejected. This limitation is enforced to somewhat narrow the 
  scope of the system and therefore the potential to mess things up and might as well be 
  removed in the future.
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
#### Supported Operators
- all arithmetic operators
- all comparison operators
- bitwise operators: and, or, xor, complement (`&`, `|`, `^`, `~`)
#### Supported functions

| function name | description                                                      |
|---------------|------------------------------------------------------------------|
| `abs`         | absolute values of a variable                                    |
| `max`         | maximum value of a variable                                      |
| `min`         | minimum value of a variable                                      |
| `mean`        | mean value of a variable                                         |
| `sum`         | sum of a variable                                                |
| `std`         | standard deviation of a variable                                 |
| `len`         | the number of values of variable                                 |
| `ismissing`   | check for missing values (nan and a possibly user defined value) |

#### Referencing Semantics
If another variable is reference within an generic test, the flags from that variable are
propagated to the checked variable.

For example:
Let `var1` and `var2` be two variables of a given dataset and `func: var1 > mean(var1)` 
the condition wheter to flag `var2`. The result of the check can be described
as `isflagged(var1) & istrue(func())`.

=======

### License
Copyright(c) 2019, 
Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ. 
All rights reserved.

The "System for Automated Quality Control" is free software. You can 
redistribute it and/or modify it under the terms of the GNU General 
Public License as published by the free Software Foundation either 
version 3 of the License, or (at your option) any later version. See the [license](license.txt) for detaily.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

