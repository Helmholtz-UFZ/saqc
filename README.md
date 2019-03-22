## Dependencies
- numpy
- pandas
- numba
- pyyaml

## Test Syntax

### Specification
A test specification contains:
- A test name, either on of the pre-defined tests or 'generic'
- Optionally a set of parametes. These should be given in
  json-object or yaml/python-dictionary style (i.e. {key: value})
- test name and parameter object/dictionary need to be seperated by comma

#### Test Parameters
- flag_period:
  + if a value is flagged, so is the given time period following the timestamp of that value
  + Number followed by a frequency specification, e.g. '5min', '6D'.
    A comprehensive list of the supported frequies can be found in the table 'Offset Aliases' in the [Pandas Docs](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects "Pandas Docs"). The (probably) most common options are also listed below:

    | frequency string | description |
    |------------------|-------------|
    | 'D'              | one day     |
    | 'H'              | one hour    |
    | 'T' or 'min'     | one minute  |
    | 'S'              | one second  |
- flag_values:
  + Number
  + if a value is flagged, so are the next n previously unflagged values

### User Defined Tests
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
#### Supported Operators
- all arithmetic operators
- all comparison operators
- bitwise operators: and, or, xor, complement (&, |, ^, ~)
#### Supported functions

| function name | description                                                      |
|---------------|------------------------------------------------------------------|
| abs           | absolute values of a variable                                    |
| max           | maximum value of a variable                                      |
| min           | minimum value of a variable                                      |
| mean          | mean value of a variable                                         |
| sum           | sum of a variable                                                |
| std           | standard deviation of a variable                                 |
| len           | the number of values of variable                                 |
| ismissing     | check for missing values (nan and a possibly user defined value) |
