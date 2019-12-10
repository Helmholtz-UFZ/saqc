# System for automated Quality Control (SaQC)

Quality Control of numerical data is an profoundly knowledge and experience based activity. Finding a robust setup is usually a time consuming and dynamic endeavor, even for an experienced
data expert.

SaQC addresses the iterative and explorative characteristics of quality control with its extensive setup and configuration possibilities and a python based extension language. To make the system flexible, many aspects of the quality
checking process, like

+ test parametrization
+ test evaluation and 
+ test exploration 

are easily configurable with plain text files.

Below its userinterface, SaQC is, thus, highly customizable and extensible. Well defined interfaces allow the extension with new quality check routines. Additionally, the core components like the flagging scheme are replaceable.

---
## Dependencies
- numpy
- pandas
- numba
- pyyaml

## Test Syntax
### Specification
- Test specifications are written in [YAML](https://en.wikipedia.org/wiki/YAML, "Wikipedia") and contain:
  + A test name, either on of the pre-defined tests or 'generic'
  + Optionally a set of parametes. These should be given in
    json-object or yaml/python-dictionary style (i.e. {key: value})
  + test name and parameter object/dictionary need to be seperated by comma
- Example: `limits, {min: 0, max: 100}`
#### Optional Test Parameters
- `flag`:
  The value to set (more precisely the value to pass to the flagging component) if the tests
  does not pass
- `flag_period`:
  + if a value is flagged, so is the given time period following the timestamp of that value
  + Number followed by a frequency specification, e.g. '5min', '6D'.
    A comprehensive list of the supported frequies can be found in the table 'Offset Aliases' in the [Pandas Docs](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects "Pandas Docs"). The (probably) most common options are also listed below:

    | frequency string | description |
    |------------------|-------------|
    | `D`              | one day     |
    | `H`              | one hour    |
    | `T` or `min`     | one minute  |
    | `S`              | one second  |
- `flag_values`:
  + Number
  + if a value is flagged, so are the next n previously unflagged values
- `assign`:
  + boolean
  + Assign the test result to a new column
### Predefined Tests

| name  | required parameters | optional parameters | description                             |
|-------|---------------------|---------------------|-----------------------------------------|
| `mad` | `z`, `length`       | `deriv = 1`         | mean absolute deviation with measure of <br> central tendency `z` and an <br> rolling window of size `length`. Optionally <br> `deriv`'s derivate of  the dataset is <br> calculated first.                       |
|       |                     |                     |                                         |
    

### User Defined Test
User defined tests allow to specify simple quality checks directly within the configuration.
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

## Contributing
### Testing
Please run the tests before you commit!
```sh
python -m pytest test
```
can save us a lot of time...

### New QC-Algorithms
Currently all test algorithms are collected within the module `funcs.functions`.
In order to make your test available for the system you need to:
- Place your code into the file `funcs/functions.py`
- Register your function by adding it to the dictionary `func_map`
  within the function body of `funcs.functions.flagDispatch`. Your function 
  will be available to the system by its key.
- Implement the common interface:
  + Function input:
    Your function needs to accept the following arguments:
    + `data: pd.DataFrame`: A dataframe holding the entire dataset (i.e. not only
       the variable, the current test is performed on)
    + `flags: pd.DataFrame`: A dataframe holding the flags for the entire 
       dataset
    + `field: String`: The name of the variable the current test is performed on
       (i.e. a column index into `data` and `columns`).
       The data and flags for this variable are available via `data[field]` and 
       `flags[field]` respectively
    + `flagger: flagger.CategoricalFlagger`: An instance of the `CategoricalFlagger` class
       (more likely one of its subclasses). To initialize, create or check
       against existing flags you should use the respective `flagger`-methods
       (`flagger.empytFlags`, `flagger.isFlagged` and `flagger.setFlag`)
    + `**kwargs: Any`: All the parameters given in the configuration file are passed
       to your function, you are of course free to make some of them required 
       by your signature. `kwargs` should be passed on to the `flagger.setFlag` 
       method, in order to allow configuration based fine tuning of the flagging
  + Function output:
    + `data: Union[np.ndarray, pd.DataFrame]`: The (hopefully unchanged) data
    + `flags: Union[np.ndarray, pd.DataFrame]`: The (most likely modified) flags
  + Note: The choosen interface allows you to not only manipulate 
    the flags, but also the data of the entire dataset within your function 
    body. This freedom might come in handy, but also requires a certain amount 
    of care to not mess things up!
  + Example: The function `flagRange` in `funcs/functions.py` may serve as an
    simple example of the general scheme

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

